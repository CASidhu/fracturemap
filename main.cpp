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
#include <cstdlib>

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
// Math Helpers (Raycasting & Surface Snap Projections)
//------------------------------------------------------------------------------
static glm::vec3 getRayFromMouse(double mouseX, double mouseY, int width, int height, const glm::mat4& view, const glm::mat4& proj) {
	float x = (2.0f * (float)mouseX) / width - 1.0f;
	float y = 1.0f - (2.0f * (float)mouseY) / height;
	glm::vec4 rayClip(x, y, -1.0f, 1.0f);
	glm::vec4 rayView = glm::inverse(proj) * rayClip;
	rayView = glm::vec4(rayView.x, rayView.y, -1.0f, 0.0f);
	return glm::normalize(glm::vec3(glm::inverse(view) * rayView));
}

static bool intersectSphere(glm::vec3 ro, glm::vec3 rd, float radius, float& t) {
	float b = 2.0f * glm::dot(ro, rd);
	float c = glm::dot(ro, ro) - radius * radius;
	float disc = b * b - 4.0f * c;
	if (disc < 0.0f) return false;
	t = (-b - std::sqrt(disc)) / 2.0f;
	return t >= 0.0f;
}

static bool intersectBox(glm::vec3 ro, glm::vec3 rd, glm::vec3 boxMin, glm::vec3 boxMax, float& t) {
	glm::vec3 invR = 1.0f / (rd + glm::vec3(1e-6f));
	glm::vec3 t0 = (boxMin - ro) * invR;
	glm::vec3 t1 = (boxMax - ro) * invR;
	glm::vec3 tmin = glm::min(t0, t1);
	glm::vec3 tmax = glm::max(t0, t1);
	float ta = std::max(tmin.x, std::max(tmin.y, tmin.z));
	float tb = std::min(tmax.x, std::min(tmax.y, tmax.z));
	if (ta <= tb && tb >= 0.0f) {
		t = ta < 0.0f ? tb : ta;
		return true;
	}
	return false;
}

static glm::vec3 projectToSurface(glm::vec3 p, int shapeType) {
	if (shapeType == 1) { // Sphere
		return glm::normalize(p) * 0.85f;
	} else if (shapeType == 0) { // Cube
		float absX = std::abs(p.x);
		float absY = std::abs(p.y);
		float absZ = std::abs(p.z);
		float maxVal = std::max({absX, absY, absZ});
		if (maxVal < 1e-4f) return p;
		if (maxVal == absX) p.x = (p.x > 0 ? 0.65f : -0.65f);
		else if (maxVal == absY) p.y = (p.y > 0 ? 0.65f : -0.65f);
		else p.z = (p.z > 0 ? 0.65f : -0.65f);
		p.x = std::clamp(p.x, -0.65f, 0.65f);
		p.y = std::clamp(p.y, -0.65f, 0.65f);
		p.z = std::clamp(p.z, -0.65f, 0.65f);
		return p;
	} else { // Torus
		float R = 0.6f, r = 0.25f;
		glm::vec2 horizontal(p.x, p.z);
		if (glm::length(horizontal) < 1e-4f) return glm::vec3(R + r, 0, 0);
		glm::vec3 coreCirclePt = glm::vec3(glm::normalize(horizontal) * R, 0.0f);
		coreCirclePt.z = coreCirclePt.y; coreCirclePt.y = 0.0f;
		return coreCirclePt + glm::normalize(p - coreCirclePt) * r;
	}
}

//------------------------------------------------------------------------------
// Geometry Generation Structures
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
		float th1 = PI * (float)i / lat, th2 = PI * (float)(i + 1) / lat;
		for (int j = 0; j < lon; j++) {
			float ph1 = 2.f * PI * (float)j / lon, ph2 = 2.f * PI * (float)(j + 1) / lon;
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
			vertex(u1, v1, p1, n1); vertex(u2, v1, p2, n2); vertex(u2, v2, p3, n3); vertex(u1, v2, p4, n4);
			m.verts.insert(m.verts.end(), {p1, p2, p3, p1, p3, p4});
			m.normals.insert(m.normals.end(), {n1, n2, n3, n1, n3, n4});
		}
	}
	return m;
}

//------------------------------------------------------------------------------
// Procedural Starburst Fracture Tree Generator
//------------------------------------------------------------------------------
static void generateBranch(std::vector<std::vector<glm::vec3>>& paths, glm::vec3 startPt, glm::vec3 dir, int steps, int shapeType, float jitter) {
	std::vector<glm::vec3> currentPath;
	currentPath.push_back(startPt);
	glm::vec3 curr = startPt;
	glm::vec3 currDir = dir;

	for (int i = 0; i < steps; ++i) {
		glm::vec3 norm = glm::normalize(curr);
		if (shapeType == 0) {
			if (std::abs(curr.x) > 0.64f) norm = glm::vec3(curr.x > 0 ? 1 : -1, 0, 0);
			else if (std::abs(curr.y) > 0.64f) norm = glm::vec3(0, curr.y > 0 ? 1 : -1, 0);
			else norm = glm::vec3(0, 0, curr.z > 0 ? 1 : -1);
		} else if (shapeType == 2) {
			float R = 0.6f;
			glm::vec2 h(curr.x, curr.z);
			glm::vec3 coreCirclePt = glm::vec3(glm::normalize(h) * R, 0.0f);
			coreCirclePt.z = coreCirclePt.y; coreCirclePt.y = 0.0f;
			norm = glm::normalize(curr - coreCirclePt);
		}

		currDir = glm::normalize(currDir - glm::dot(currDir, norm) * norm);

		float r1 = (float)((double)rand() / (double)RAND_MAX) * 2.0f - 1.0f;
		glm::vec3 sideDir = glm::normalize(glm::cross(currDir, norm));
		currDir = glm::normalize(currDir + sideDir * r1 * 0.35f * jitter);

		glm::vec3 nextPt = curr + currDir * 0.035f;
		nextPt = projectToSurface(nextPt, shapeType);

		currentPath.push_back(nextPt);
		curr = nextPt;

		if (i > 2 && i < steps - 3 && ((double)rand() / (double)RAND_MAX) < 0.12 && paths.size() < 16) {
			float splitAngle = (((double)rand() / (double)RAND_MAX) > 0.5 ? 1.0f : -1.0f) * 0.6f;
			glm::vec3 subDir = glm::normalize(currDir * std::cos(splitAngle) + sideDir * std::sin(splitAngle));
			generateBranch(paths, curr, subDir, steps - i, shapeType, jitter);
		}
	}
	paths.push_back(currentPath);
}

static Mesh makeStarburstCracks(glm::vec3 hitPoint, int numMainArms, int shapeType, float width, float depth, float jitter) {
	Mesh m;
	std::vector<std::vector<glm::vec3>> paths;
	constexpr float TWO_PI = 6.28318530718f;

	glm::vec3 norm = glm::normalize(hitPoint);
	if (shapeType == 0) {
		if (std::abs(hitPoint.x) > 0.64f) norm = glm::vec3(hitPoint.x > 0 ? 1 : -1, 0, 0);
		else if (std::abs(hitPoint.y) > 0.64f) norm = glm::vec3(0, hitPoint.y > 0 ? 1 : -1, 0);
		else norm = glm::vec3(0, 0, hitPoint.z > 0 ? 1 : -1);
	}
	glm::vec3 helper = std::abs(norm.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
	glm::vec3 tangentX = glm::normalize(glm::cross(norm, helper));
	glm::vec3 tangentY = glm::cross(norm, tangentX);

	for (int i = 0; i < numMainArms; ++i) {
		float angle = (i * TWO_PI) / numMainArms;
		glm::vec3 armDir = glm::normalize(tangentX * std::cos(angle) + tangentY * std::sin(angle));
		generateBranch(paths, hitPoint, armDir, 25, shapeType, jitter);
	}

	for (const auto& path : paths) {
		if (path.size() < 2) continue;
		std::vector<glm::vec3> leftSide, rightSide, valleyFloor;

		for (size_t i = 0; i < path.size(); ++i) {
			glm::vec3 p = path[i];
			glm::vec3 n = glm::normalize(p);
			if (shapeType == 0) {
				if (std::abs(p.x) > 0.64f) n = glm::vec3(p.x > 0 ? 1 : -1, 0, 0);
				else if (std::abs(p.y) > 0.64f) n = glm::vec3(0, p.y > 0 ? 1 : -1, 0);
				else n = glm::vec3(0, 0, p.z > 0 ? 1 : -1);
			}
			glm::vec3 forward = (i == 0) ? glm::normalize(path[i + 1] - p) : glm::normalize(p - path[i - 1]);
			glm::vec3 t = glm::normalize(glm::cross(forward, n));

			leftSide.push_back(p - t * (width * 0.5f) + n * 0.002f);
			rightSide.push_back(p + t * (width * 0.5f) + n * 0.002f);
			valleyFloor.push_back(p - n * depth);
		}

		for (size_t i = 0; i < path.size() - 1; ++i) {
			m.verts.insert(m.verts.end(), {leftSide[i], valleyFloor[i], leftSide[i+1], valleyFloor[i], valleyFloor[i+1], leftSide[i+1]});
			glm::vec3 nl = glm::normalize(glm::cross(valleyFloor[i] - leftSide[i], leftSide[i+1] - leftSide[i]));
			m.normals.insert(m.normals.end(), {nl, nl, nl, nl, nl, nl});

			m.verts.insert(m.verts.end(), {valleyFloor[i], rightSide[i], valleyFloor[i+1], rightSide[i], rightSide[i+1], valleyFloor[i+1]});
			glm::vec3 nr = glm::normalize(glm::cross(rightSide[i+1] - rightSide[i], valleyFloor[i] - rightSide[i]));
			m.normals.insert(m.normals.end(), {nr, nr, nr, nr, nr, nr});
		}
	}
	return m;
}

//------------------------------------------------------------------------------
// Shader Setup
//------------------------------------------------------------------------------
static const char* VERT_SRC = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 M; uniform mat4 V; uniform mat4 P;
out vec3 vNormal; out vec3 vFragPos;
void main() {
	vFragPos = vec3(M * vec4(aPos, 1.0));
	vNormal = mat3(transpose(inverse(M))) * aNormal;
	gl_Position = P * V * M * vec4(aPos, 1.0);
}
)";

static const char* FRAG_SRC = R"(#version 300 es
precision highp float;
in vec3 vNormal; in vec3 vFragPos;
uniform vec3 lightPos; uniform vec3 lightColor; uniform vec3 diffuseColor; uniform float ambientStrength;
out vec4 fragColor;
void main() {
	vec3 norm = normalize(vNormal);
	vec3 lightDir = normalize(lightPos - vFragPos);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 result = lightColor * (ambientStrength + diff) * diffuseColor;
	fragColor = vec4(result, 1.0);
}
)";

//------------------------------------------------------------------------------
// Shader Utilities
//------------------------------------------------------------------------------
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
	if (g.count == 0) return g;

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
// Application Configuration Loop
//------------------------------------------------------------------------------
struct App {
	GLFWwindow* window = nullptr;
	Camera camera{glm::radians(45.f), glm::radians(35.f), 3.2f}; // Default pitch at 45 degrees

	GLuint program = 0;
	GPUMesh hostMeshes[3];
	GPUMesh crackMesh;
	int shapeIndex = 2; // Default to Torus (0=Cube, 1=Sphere, 2=Torus)

	bool wireframe = true; // Default to wireframe enabled
	bool autoRotate = false;
	bool dragging = false;
	double lastX = 0.0, lastY = 0.0;

	glm::vec3 lightPos{2.f, 2.f, 2.f};
	glm::vec3 lightColor{1.f, 1.f, 1.f};
	glm::vec3 diffuseColor{0.6f, 0.5f, 0.4f};
	float ambient = 0.2f;

	int mainArms = 5;
	float crackWidth = 0.03f;
	float crackDepth = 0.02f;
	float crackJitter = 1.0f;
	glm::vec3 crackColor{0.08f, 0.08f, 0.08f};
	glm::vec3 activeStrikePoint{0.0f, 0.6f, 0.25f}; // Snapped coordinate component initialization on Torus rim

	void updateCrackGeometry() {
		crackMesh.clear();
		Mesh m = makeStarburstCracks(activeStrikePoint, mainArms, shapeIndex, crackWidth, crackDepth, crackJitter);
		crackMesh = uploadMesh(m);
	}
};

static void mouseButtonCB(GLFWwindow* w, int button, int action, int /*mods*/) {
	if (ImGui::GetIO().WantCaptureMouse) return;
	App* app = (App*)glfwGetWindowUserPointer(w);
	
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		app->dragging = (action == GLFW_PRESS);
	} 
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		double mx, my;
		glfwGetCursorPos(w, &mx, &my);
		int width, height;
		glfwGetFramebufferSize(w, &width, &height);

		glm::mat4 V = app->camera.getView();
		glm::mat4 P = glm::perspective(glm::radians(45.f), height > 0 ? (float)width / (float)height : 1.f, 0.1f, 100.f);

		glm::vec3 ro = app->camera.getPos();
		glm::vec3 rd = getRayFromMouse(mx, my, width, height, V, P);

		float t = -1.0f;
		bool hit = false;

		if (app->shapeIndex == 1)      hit = intersectSphere(ro, rd, 0.85f, t);
		else if (app->shapeIndex == 0) hit = intersectBox(ro, rd, glm::vec3(-0.65f), glm::vec3(0.65f), t);
		else                           hit = intersectSphere(ro, rd, 0.75f, t); 

		if (hit && t >= 0.0f) {
			glm::vec3 hitPoint = ro + t * rd;
			app->activeStrikePoint = projectToSurface(hitPoint, app->shapeIndex);
			app->updateCrackGeometry();
		}
	}
}

static void cursorPosCB(GLFWwindow* w, double x, double y) {
	App* app = (App*)glfwGetWindowUserPointer(w);
	if (app->dragging && !ImGui::GetIO().WantCaptureMouse) {
		app->camera.incrementTheta((float)(y - app->lastY));
		app->camera.incrementPhi((float)(x - app->lastX));
	}
	app->lastX = x; app->lastY = y;
}

static void scrollCB(GLFWwindow* w, double /*xoff*/, double yoff) {
	if (ImGui::GetIO().WantCaptureMouse) return;
	App* app = (App*)glfwGetWindowUserPointer(w);
	app->camera.incrementR((float)yoff * 0.2f);
}

static void frame(void* arg) {
	App& app = *(App*)arg;
	glfwPollEvents();

	ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

	ImGui::Begin("Fracture Workshop");
	const char* shapes[] = {"Cube", "Sphere", "Torus"};
	if (ImGui::Combo("Primitive Target", &app.shapeIndex, shapes, 3)) {
		app.activeStrikePoint = projectToSurface(glm::vec3(0.0f, 1.0f, 0.0f), app.shapeIndex);
		app.updateCrackGeometry();
	}
	ImGui::Checkbox("Wireframe Mode", &app.wireframe);
	ImGui::Checkbox("Continuous Orbit Rotation", &app.autoRotate);
	ImGui::ColorEdit3("Object Tint", &app.diffuseColor[0]);
	
	ImGui::Spacing(); ImGui::Separator();
	ImGui::Text("Procedural Starburst Generator");
	if (ImGui::SliderInt("Primary Arms", &app.mainArms, 2, 10) ||
		ImGui::SliderFloat("Flank Width", &app.crackWidth, 0.01f, 0.12f) ||
		ImGui::SliderFloat("Valley Depth", &app.crackDepth, 0.005f, 0.08f) ||
		ImGui::SliderFloat("Branch Jitter", &app.crackJitter, 0.0f, 2.5f)) {
		app.updateCrackGeometry();
	}
	ImGui::ColorEdit3("Indentation Tint", &app.crackColor[0]);
	ImGui::Spacing();
	ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "INSTRUCTION:");
	ImGui::TextWrapped("Right-Click directly on the object's surface to trigger a multi-arm branching fracture strike.");
	ImGui::End();

	if (app.autoRotate) app.camera.incrementPhi(-0.4f);

	int w, h; glfwGetFramebufferSize(app.window, &w, &h);
	glViewport(0, 0, w, h);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
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

	// Base Shape
	glUniform3fv(glGetUniformLocation(app.program, "diffuseColor"), 1, glm::value_ptr(app.diffuseColor));
	GPUMesh& hostMesh = app.hostMeshes[app.shapeIndex];
	glBindVertexArray(hostMesh.vao);
	if (app.wireframe) glDrawElements(GL_LINES, hostMesh.wireCount, GL_UNSIGNED_INT, (void*)0);
	else glDrawArrays(GL_TRIANGLES, 0, hostMesh.count);

	// Branching Fractures
	if (app.crackMesh.count > 0) {
		glUniform3fv(glGetUniformLocation(app.program, "diffuseColor"), 1, glm::value_ptr(app.crackColor));
		glBindVertexArray(app.crackMesh.vao);
		if (app.wireframe) glDrawElements(GL_LINES, app.crackMesh.wireCount, GL_UNSIGNED_INT, (void*)0);
		else glDrawArrays(GL_TRIANGLES, 0, app.crackMesh.count);
	}

	ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	glfwSwapBuffers(app.window);
}

int main() {
	if (!glfwInit()) return 1;
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	static App app;
	app.window = glfwCreateWindow(900, 700, "Starburst Crack Engine", nullptr, nullptr);
	if (!app.window) return 1;

	glfwMakeContextCurrent(app.window); glfwSetWindowUserPointer(app.window, &app);
	glfwSetMouseButtonCallback(app.window, mouseButtonCB);
	glfwSetCursorPosCallback(app.window, cursorPosCB);
	glfwSetScrollCallback(app.window, scrollCB);

	IMGUI_CHECKVERSION(); ImGui::CreateContext(); ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(app.window, true); ImGui_ImplOpenGL3_Init("#version 300 es");

	app.program = linkProgram(VERT_SRC, FRAG_SRC);
	app.hostMeshes[0] = uploadMesh(makeCube());
	app.hostMeshes[1] = uploadMesh(makeSphere());
	app.hostMeshes[2] = uploadMesh(makeTorus());

	app.updateCrackGeometry();
	emscripten_set_main_loop_arg(frame, &app, 0, 1);
	return 0;
}
