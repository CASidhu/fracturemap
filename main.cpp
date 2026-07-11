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
#include <algorithm>

#ifdef __EMSCRIPTEN__
extern "C" {
    int glfwGetError(const char** description) {
        if (description) *description = nullptr;
        return 0;
    }

    int glfwGetGamepadState(int jid, GLFWgamepadstate* state) {
        return 0;
    }
}
#endif

//------------------------------------------------------------------------------
// Camera
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
		constexpr float limit = 1.5533f;
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
// Geometry Structures
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
// Procedural Crack Generation Engine (Phenomenological Sweep Approach)
//------------------------------------------------------------------------------
static Mesh makeCrackModel(int shapeType, float crackWidth, float crackDepth, float noiseScale) {
	Mesh m;
	std::vector<glm::vec3> path;
	constexpr float PI = 3.14159265358979f;

	// 1. Generate perturbed 2D/3D crack trajectory skeleton based on the host shape
	if (shapeType == 1) { // Sphere trajectory
		float radius = 0.852f; // Sits slightly above host surface to prevent Z-fighting
		int steps = 60;
		float startPhi = 0.0f;
		for (int i = 0; i < steps; ++i) {
			float t = (float)i / (steps - 1);
			float theta = PI * 0.3f + t * (PI * 0.4f); // Traverses mid-latitudes
			float phi = startPhi + std::sin(t * PI * 2.5f) * 0.15f * noiseScale;
			
			glm::vec3 p(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
			path.push_back(p * radius);
			startPhi += 0.04f;
		}
	} else if (shapeType == 0) { // Cube trajectory (Face-aligned segment layout)
		int steps = 40;
		float s = 0.652f;
		for (int i = 0; i < steps; ++i) {
			float t = (float)i / (steps - 1);
			float x = -s + t * (s * 2.0f);
			float y = s;
			float z = std::sin(t * PI * 3.0f) * 0.12f * noiseScale;
			path.push_back(glm::vec3(x, y, z));
		}
	} else { // Torus trajectory
		int steps = 80;
		float R = 0.6f, r = 0.252f;
		for (int i = 0; i < steps; ++i) {
			float t = (float)i / (steps - 1);
			float u = t * (PI * 1.5f);
			float v = std::sin(t * PI * 6.0f) * 0.4f * noiseScale;
			
			float cu = std::cos(u), su = std::sin(u);
			float cv = std::cos(v), sv = std::sin(v);
			path.push_back(glm::vec3((R + r * cv) * cu, r * sv, (R + r * cv) * su));
		}
	}

	if (path.size() < 2) return m;

	// 2. Sweep Profile Cross-Section (V-shaped fracture valley carving logic)
	std::vector<glm::vec3> leftSide, rightSide, valleyFloor;

	for (size_t i = 0; i < path.size(); ++i) {
		glm::vec3 p = path[i];
		glm::vec3 normal = glm::normalize(p); // Default spherical projection mapping
		if (shapeType == 0) normal = glm::vec3(0.0f, 1.0f, 0.0f); // Top face normal flat projection

		glm::vec3 forward;
		if (i == 0) forward = glm::normalize(path[i + 1] - p);
		else if (i == path.size() - 1) forward = glm::normalize(p - path[i - 1]);
		else forward = glm::normalize(path[i + 1] - path[i - 1]);

		glm::vec3 tangent = glm::normalize(glm::cross(forward, normal));

		leftSide.push_back(p - tangent * (crackWidth * 0.5f));
		rightSide.push_back(p + tangent * (crackWidth * 0.5f));
		valleyFloor.push_back(p - normal * crackDepth); // Carves inward relative to surface normal
	}

	// 3. Tessellate the generated swept cross-section points into a continuous triangle strip
	for (size_t i = 0; i < path.size() - 1; ++i) {
		glm::vec3 l0 = leftSide[i],    l1 = leftSide[i + 1];
		glm::vec3 r0 = rightSide[i],   r1 = rightSide[i + 1];
		glm::vec3 v0 = valleyFloor[i], v1 = valleyFloor[i + 1];

		// Left flank of the crack valley
		m.verts.insert(m.verts.end(), {l0, v0, l1, v0, v1, l1});
		glm::vec3 nl = glm::normalize(glm::cross(v0 - l0, l1 - l0));
		m.normals.insert(m.normals.end(), {nl, nl, nl, nl, nl, nl});

		// Right flank of the crack valley
		m.verts.insert(m.verts.end(), {v0, r0, v1, r0, r1, v1});
		glm::vec3 nr = glm::normalize(glm::cross(r1 - r0, v0 - r0));
		m.normals.insert(m.normals.end(), {nr, nr, nr, nr, nr, nr});
	}

	return m;
}

//------------------------------------------------------------------------------
// GPU upload
//------------------------------------------------------------------------------
struct GPUMesh {
	GLuint vao = 0, vboPos = 0, vboNorm = 0, ebo = 0;
	GLsizei count = 0;
	GLsizei wireCount = 0;

	void clear() {
		if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
		if (vboPos) { glDeleteBuffers(1, &vboPos); vboPos = 0; }
		if (vboNorm) { glDeleteBuffers(1, &vboNorm); vboNorm = 0; }
		if (ebo) { glDeleteBuffers(1, &ebo); ebo = 0; }
		count = 0; wireCount = 0;
	}
};

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
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, wireIdx.size() * sizeof(GLuint), wireIdx.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	return g;
}

//------------------------------------------------------------------------------
// Shaders
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
// App State
//------------------------------------------------------------------------------
struct App {
	GLFWwindow* window = nullptr;
	Camera camera{glm::radians(20.f), glm::radians(35.f), 3.2f};

	GLuint program = 0;
	GPUMesh hostMeshes[3];
	GPUMesh crackMesh;
	int shapeIndex = 1;

	bool wireframe = false;
	bool autoRotate = true;
	bool dragging = false;
	double lastX = 0.0, lastY = 0.0;

	glm::vec3 lightPos{2.f, 2.f, 2.f};
	glm::vec3 lightColor{1.f, 1.f, 1.f};
	glm::vec3 diffuseColor{0.85f, 0.2f, 0.25f};
	float ambient = 0.15f;

	// Crack parameters
	bool showCracks = true;
	float crackWidth = 0.04f;
	float crackDepth = 0.03f;
	float crackJitter = 1.0f;
	glm::vec3 crackColor{0.08f, 0.08f, 0.08f}; // Dark indentation shadow signature

	void updateCrackGeometry() {
		crackMesh.clear();
		Mesh m = makeCrackModel(shapeIndex, crackWidth, crackDepth, crackJitter);
		if (m.verts.size() > 0) {
			crackMesh = uploadMesh(m);
		}
	}
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
// Per-Frame Loop Callback
//------------------------------------------------------------------------------
static void frame(void* arg) {
	App& app = *(App*)arg;

	glfwPollEvents();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Controls");
	const char* shapes[] = {"Cube", "Sphere", "Torus"};
	if (ImGui::Combo("Host Shape", &app.shapeIndex, shapes, 3)) {
		app.updateCrackGeometry();
	}
	ImGui::Checkbox("Wireframe View", &app.wireframe);
	ImGui::Checkbox("Auto-rotate Base", &app.autoRotate);
	ImGui::ColorEdit3("Object Base Colour", &app.diffuseColor[0]);
	ImGui::SliderFloat("Ambient Intensity", &app.ambient, 0.0f, 1.0f);
	
	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Text("Fracture Pattern Configuration");
	if (ImGui::Checkbox("Show Surface Cracks", &app.showCracks)) {
		app.updateCrackGeometry();
	}
	if (app.showCracks) {
		if (ImGui::SliderFloat("Crack Width", &app.crackWidth, 0.005f, 0.15f) ||
			ImGui::SliderFloat("Crack Depth", &app.crackDepth, 0.005f, 0.10f) ||
			ImGui::SliderFloat("Pattern Jitter", &app.crackJitter, 0.0f, 3.0f)) {
			app.updateCrackGeometry();
		}
		ImGui::ColorEdit3("Fracture Valley Color", &app.crackColor[0]);
	}
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
	glUniform1f(glGetUniformLocation(app.program, "ambientStrength"), app.ambient);

	// Render Base Primitive
	glUniform3fv(glGetUniformLocation(app.program, "diffuseColor"), 1, glm::value_ptr(app.diffuseColor));
	GPUMesh& mesh = app.hostMeshes[app.shapeIndex];
	glBindVertexArray(mesh.vao);
	if (app.wireframe) {
		glDrawElements(GL_LINES, mesh.wireCount, GL_UNSIGNED_INT, (void*)0);
	} else {
		glDrawArrays(GL_TRIANGLES, 0, mesh.count);
	}

	// Render Procedural Fracture Mesh Overlays
	if (app.showCracks && app.crackMesh.count > 0) {
		glUniform3fv(glGetUniformLocation(app.program, "diffuseColor"), 1, glm::value_ptr(app.crackColor));
		glBindVertexArray(app.crackMesh.vao);
		if (app.wireframe) {
			glDrawElements(GL_LINES, app.crackMesh.wireCount, GL_UNSIGNED_INT, (void*)0);
		} else {
			glDrawArrays(GL_TRIANGLES, 0, app.crackMesh.count);
		}
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

	static App app;

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
	app.hostMeshes[0] = uploadMesh(makeCube());
	app.hostMeshes[1] = uploadMesh(makeSphere());
	app.hostMeshes[2] = uploadMesh(makeTorus());

	app.updateCrackGeometry();

	emscripten_set_main_loop_arg(frame, &app, 0, 1);
	return 0;
}
