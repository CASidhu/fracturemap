// Minimal WebGL2 (via Emscripten) demo derived from the original desktop
// OpenGL skeleton. The OBJ loader and mcut boolean-operation pipeline have
// been replaced with procedurally generated shapes so the whole thing can
// run with zero asset files and no native-only dependencies.

#include <GLES3/gl3.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <emscripten.h>

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <GLFW/glfw3.h>

extern "C" {
    int glfwGetError(const char** description) {
        if (description) *description = nullptr;
        return 0; // Returns no error
    }

    int glfwGetGamepadState(int jid, GLFWgamepadstate* state) {
        return 0; // Returns GLFW_FALSE
    }
}
#endif

//------------------------------------------------------------------------------
// Camera (ported from Camera.h/.cpp, trimmed)
//------------------------------------------------------------------------------
struct Camera {
	float theta, phi, radius;

	Camera(float t, float p, float r) : theta(t), phi(p), radius(r) {}

	glm::vec3 getPos() const {
		return radius * glm::vec3(std::cos(theta) * std::sin(phi), std::sin(theta), std::cos(theta) * std::cos(phi));
	}

	glm::mat4 getView() const {
		return glm::lookAt(getPos(), glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
	}

	void incrementTheta(float dt) {
		float next = theta + dt / 100.f;
		constexpr float limit = 1.5533f; // just under pi/2 to avoid gimbal flip
		if (next < limit && next > -limit) theta = next;
	}

	void incrementPhi(float dp) {
		phi -= dp / 100.f;
		constexpr float twoPi = 6.28318530718f;
		if (phi > twoPi) phi -= twoPi;
		else if (phi < 0.f) phi += twoPi;
	}

	void incrementR(float dr) {
		radius = std::max(0.75f, radius - dr);
	}
};

//------------------------------------------------------------------------------
// Procedural geometry (replaces GeomLoaderForOBJ + HEGeometry for this demo)
//------------------------------------------------------------------------------
struct Mesh {
	std::vector<glm::vec3> verts;
	std::vector<glm::vec3> normals;
};

static Mesh makeCube() {
	Mesh m;
	auto face = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 n) {
		m.verts.insert(m.verts.end(), {a, b, c, a, c, d});
		m.normals.insert(m.normals.end(), {n, n, n, n, n, n});
	};

	const float s = 0.65f;
	glm::vec3 p000(-s, -s, -s), p100(s, -s, -s), p110(s, s, -s), p010(-s, s, -s);
	glm::vec3 p001(-s, -s, s), p101(s, -s, s), p111(s, s, s), p011(-s, s, s);

	face(p000, p010, p110, p100, glm::vec3(0, 0, -1));
	face(p101, p111, p011, p001, glm::vec3(0, 0, 1));
	face(p001, p011, p010, p000, glm::vec3(-1, 0, 0));
	face(p100, p110, p111, p101, glm::vec3(1, 0, 0));
	face(p000, p100, p101, p001, glm::vec3(0, -1, 0));
	face(p010, p011, p111, p110, glm::vec3(0, 1, 0));

	return m;
}

static Mesh makeSphere(int lat = 28, int lon = 28, float radius = 0.85f) {
	Mesh m;
	constexpr float PI = 3.14159265358979f;

	auto point = [&](float theta, float phi) {
		return glm::vec3(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
	};

	for (int i = 0; i < lat; i++) {
		float th1 = PI * (float)i / lat;
		float th2 = PI * (float)(i + 1) / lat;
		for (int j = 0; j < lon; j++) {
			float ph1 = 2.f * PI * (float)j / lon;
			float ph2 = 2.f * PI * (float)(j + 1) / lon;

			glm::vec3 n1 = point(th1, ph1), n2 = point(th2, ph1), n3 = point(th2, ph2), n4 = point(th1, ph2);

			m.verts.insert(m.verts.end(), {radius * n1, radius * n2, radius * n3, radius * n1, radius * n3, radius * n4});
			m.normals.insert(m.normals.end(), {n1, n2, n3, n1, n3, n4});
		}
	}
	return m;
}

static Mesh makeTorus(int rings = 40, int sides = 20, float R = 0.6f, float r = 0.25f) {
	Mesh m;
	constexpr float TWO_PI = 6.28318530718f;

	auto vertex = [&](float u, float v, glm::vec3& pos, glm::vec3& normal) {
		float cu = std::cos(u), su = std::sin(u);
		float cv = std::cos(v), sv = std::sin(v);
		pos = glm::vec3((R + r * cv) * cu, r * sv, (R + r * cv) * su);
		normal = glm::vec3(cv * cu, sv, cv * su);
	};

	for (int i = 0; i < rings; i++) {
		float u1 = TWO_PI * i / rings, u2 = TWO_PI * (i + 1) / rings;
		for (int j = 0; j < sides; j++) {
			float v1 = TWO_PI * j / sides, v2 = TWO_PI * (j + 1) / sides;

			glm::vec3 p1, n1, p2, n2, p3, n3, p4, n4;
			vertex(u1, v1, p1, n1);
			vertex(u2, v1, p2, n2);
			vertex(u2, v2, p3, n3);
			vertex(u1, v2, p4, n4);

			m.verts.insert(m.verts.end(), {p1, p2, p3, p1, p3, p4});
			m.normals.insert(m.normals.end(), {n1, n2, n3, n1, n3, n4});
		}
	}
	return m;
}

//------------------------------------------------------------------------------
// GPU upload (replaces GPU_Geometry/VertexArray/VertexBuffer for this demo)
//------------------------------------------------------------------------------
struct GPUMesh {
	GLuint vao = 0, vboPos = 0, vboNorm = 0, ebo = 0;
	GLsizei count = 0;     // vertex count for the GL_TRIANGLES draw
	GLsizei wireCount = 0; // index count for the wireframe GL_LINES draw
};

// WebGL2/GLES3 has no glPolygonMode(GL_LINE) (that's desktop-GL-only), so
// "wireframe" is implemented as an actual GL_LINES draw over a per-triangle
// edge index buffer, built alongside the normal triangle-list buffers.
static GPUMesh uploadMesh(const Mesh& m) {
	GPUMesh g;
	g.count = (GLsizei)m.verts.size();

	glGenVertexArrays(1, &g.vao);
	glBindVertexArray(g.vao);

	glGenBuffers(1, &g.vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, g.vboPos);
	glBufferData(GL_ARRAY_BUFFER, m.verts.size() * sizeof(glm::vec3), m.verts.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glGenBuffers(1, &g.vboNorm);
	glBindBuffer(GL_ARRAY_BUFFER, g.vboNorm);
	glBufferData(GL_ARRAY_BUFFER, m.normals.size() * sizeof(glm::vec3), m.normals.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	std::vector<GLuint> wireIdx;
	wireIdx.reserve(m.verts.size() * 2);
	for (GLuint i = 0; i + 2 < m.verts.size(); i += 3) {
		wireIdx.push_back(i);     wireIdx.push_back(i + 1);
		wireIdx.push_back(i + 1); wireIdx.push_back(i + 2);
		wireIdx.push_back(i + 2); wireIdx.push_back(i);
	}
	g.wireCount = (GLsizei)wireIdx.size();

	glGenBuffers(1, &g.ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo); // captured into the VAO's element-array binding
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, wireIdx.size() * sizeof(GLuint), wireIdx.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	return g;
}

//------------------------------------------------------------------------------
// Shaders (GLSL ES 300, i.e. WebGL2) - adapted from test.vert / test.frag
//------------------------------------------------------------------------------
static const char* VERT_SRC = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
	vFragPos = vec3(M * vec4(aPos, 1.0));
	vNormal = mat3(transpose(inverse(M))) * aNormal;
	gl_Position = P * V * M * vec4(aPos, 1.0);
}
)";

static const char* FRAG_SRC = R"(#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 diffuseColor;
uniform float ambientStrength;

out vec4 fragColor;

void main() {
	vec3 norm = normalize(vNormal);
	vec3 lightDir = normalize(lightPos - vFragPos);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 result = lightColor * (ambientStrength + diff) * diffuseColor;
	fragColor = vec4(result, 1.0);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetShaderInfoLog(s, sizeof(log), nullptr, log);
		printf("[shader compile error]\n%s\n", log);
	}
	return s;
}

static GLuint linkProgram(const char* vsrc, const char* fsrc) {
	GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
	GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetProgramInfoLog(p, sizeof(log), nullptr, log);
		printf("[program link error]\n%s\n", log);
	}
	glDeleteShader(vs);
	glDeleteShader(fs);
	return p;
}

//------------------------------------------------------------------------------
// App state + input handling
//------------------------------------------------------------------------------
struct App {
	GLFWwindow* window = nullptr;
	Camera camera{glm::radians(20.f), glm::radians(35.f), 3.2f};

	GLuint program = 0;
	GPUMesh meshes[3];
	int shapeIndex = 1; // 0 = cube, 1 = sphere, 2 = torus

	bool wireframe = false;
	bool autoRotate = true;
	bool dragging = false;
	double lastX = 0.0, lastY = 0.0;

	glm::vec3 lightPos{2.f, 2.f, 2.f};
	glm::vec3 lightColor{1.f, 1.f, 1.f};
	glm::vec3 diffuseColor{0.85f, 0.2f, 0.25f};
	float ambient = 0.15f;
};

static void mouseButtonCB(GLFWwindow* w, int button, int action, int /*mods*/) {
	if (ImGui::GetIO().WantCaptureMouse) return;
	App* app = (App*)glfwGetWindowUserPointer(w);
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		app->dragging = (action == GLFW_PRESS);
	}
}

static void cursorPosCB(GLFWwindow* w, double x, double y) {
	App* app = (App*)glfwGetWindowUserPointer(w);
	if (app->dragging && !ImGui::GetIO().WantCaptureMouse) {
		app->camera.incrementTheta((float)(y - app->lastY));
		app->camera.incrementPhi((float)(x - app->lastX));
	}
	app->lastX = x;
	app->lastY = y;
}

static void scrollCB(GLFWwindow* w, double /*xoff*/, double yoff) {
	if (ImGui::GetIO().WantCaptureMouse) return;
	App* app = (App*)glfwGetWindowUserPointer(w);
	app->camera.incrementR((float)yoff * 0.2f);
}

//------------------------------------------------------------------------------
// Per-frame callback, driven by the browser via emscripten_set_main_loop_arg
//------------------------------------------------------------------------------
static void frame(void* arg) {
	App& app = *(App*)arg;

	glfwPollEvents();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Controls");
	const char* shapes[] = {"Cube", "Sphere", "Torus"};
	ImGui::Combo("Shape", &app.shapeIndex, shapes, 3);
	ImGui::Checkbox("Wireframe", &app.wireframe);
	ImGui::Checkbox("Auto-rotate", &app.autoRotate);
	ImGui::ColorEdit3("Object colour", &app.diffuseColor[0]);
	ImGui::ColorEdit3("Light colour", &app.lightColor[0]);
	ImGui::DragFloat3("Light position", &app.lightPos[0], 0.1f);
	ImGui::SliderFloat("Ambient", &app.ambient, 0.0f, 1.0f);
	ImGui::Separator();
	ImGui::TextWrapped("Drag with left mouse button to orbit. Scroll to zoom.");
	ImGui::End();

	if (app.autoRotate) app.camera.incrementPhi(-0.6f);

	int w, h;
	glfwGetFramebufferSize(app.window, &w, &h);
	glViewport(0, 0, w, h);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.11f, 0.11f, 0.13f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(app.program);

	glm::mat4 M(1.0f);
	glm::mat4 V = app.camera.getView();
	glm::mat4 P = glm::perspective(glm::radians(45.f), h > 0 ? (float)w / (float)h : 1.f, 0.1f, 100.f);

	glUniformMatrix4fv(glGetUniformLocation(app.program, "M"), 1, GL_FALSE, glm::value_ptr(M));
	glUniformMatrix4fv(glGetUniformLocation(app.program, "V"), 1, GL_FALSE, glm::value_ptr(V));
	glUniformMatrix4fv(glGetUniformLocation(app.program, "P"), 1, GL_FALSE, glm::value_ptr(P));
	glUniform3fv(glGetUniformLocation(app.program, "lightPos"), 1, glm::value_ptr(app.lightPos));
	glUniform3fv(glGetUniformLocation(app.program, "lightColor"), 1, glm::value_ptr(app.lightColor));
	glUniform3fv(glGetUniformLocation(app.program, "diffuseColor"), 1, glm::value_ptr(app.diffuseColor));
	glUniform1f(glGetUniformLocation(app.program, "ambientStrength"), app.ambient);

	GPUMesh& mesh = app.meshes[app.shapeIndex];
	glBindVertexArray(mesh.vao);
	if (app.wireframe) {
		glDrawElements(GL_LINES, mesh.wireCount, GL_UNSIGNED_INT, (void*)0);
	} else {
		glDrawArrays(GL_TRIANGLES, 0, mesh.count);
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(app.window);
}

int main() {
	if (!glfwInit()) {
		printf("glfwInit failed\n");
		return 1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	static App app; // static: simplest way to give it a stable address for the JS-driven main loop

	app.window = glfwCreateWindow(900, 700, "Geometry Demo", nullptr, nullptr);
	if (!app.window) {
		printf("glfwCreateWindow failed\n");
		return 1;
	}
	glfwMakeContextCurrent(app.window);
	glfwSetWindowUserPointer(app.window, &app);
	glfwSetMouseButtonCallback(app.window, mouseButtonCB);
	glfwSetCursorPosCallback(app.window, cursorPosCB);
	glfwSetScrollCallback(app.window, scrollCB);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(app.window, true);
	ImGui_ImplOpenGL3_Init("#version 300 es");

	app.program = linkProgram(VERT_SRC, FRAG_SRC);
	app.meshes[0] = uploadMesh(makeCube());
	app.meshes[1] = uploadMesh(makeSphere());
	app.meshes[2] = uploadMesh(makeTorus());

	emscripten_set_main_loop_arg(frame, &app, 0, 1);
	return 0;
}
