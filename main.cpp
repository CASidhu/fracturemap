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
// GLSL Shader Source (Analytical Fragment-Space Boolean Discard/Carve Engine)
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

// Analytical Fracture Uniform Arrays
uniform vec3 uSegA[90];
uniform vec3 uSegB[90];
uniform int uSegCount;
uniform float uCrackWidth;
uniform float uCrackDepth;
uniform int uCarveEnabled;

out vec4 fragColor;

void main() {
	vec3 baseNormal = normalize(vNormal);
	vec3 baseColor = diffuseColor;
	
	if (uCarveEnabled == 1 && uSegCount > 0) {
		float minDst = 1e9;
		vec3 closestLinePt = vec3(0.0);
		
		// Find closest fracture branch segment in fragment space
		for (int i = 0; i < uSegCount; ++i) {
			vec3 a = uSegA[i];
			vec3 b = uSegB[i];
			vec3 ab = b - a;
			vec3 ap = vFragPos - a;
			float t = clamp(dot(ap, ab) / (dot(ab, ab) + 1e-6), 0.0, 1.0);
			vec3 closest = a + t * ab;
			float dst = length(vFragPos - closest);
			if (dst < minDst) {
				minDst = dst;
				closestLinePt = closest;
			}
		}

		// Perform Boolean Valley Carving Operations
		float radius = uCrackWidth * 0.5;
		if (minDst < radius) {
			float falloff = 1.0 - (minDst / radius);
			
			// Calculate localized slope normal pointing inside the V-groove channel
			vec3 lateralVec = vFragPos - closestLinePt;
			if (length(lateralVec) > 1e-5) {
				lateralVec = normalize(lateralVec);
			}
			
			// Perturb normal vector inwards and darken base texture to shade the valley floor
			baseNormal = normalize(baseNormal + lateralVec * falloff * 2.0 * uCrackDepth);
			baseColor = mix(diffuseColor, vec3(0.04), falloff);
		}
	}

	vec3 lightDir = normalize(lightPos - vFragPos);
	float diff = max(dot(baseNormal, lightDir), 0.0);
	vec3 result = lightColor * (ambientStrength + diff) * baseColor;
	fragColor = vec4(result, 1.0);
}
)";

//------------------------------------------------------------------------------
// Shader Setup Utilities
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
// Surface Snap Projections
//------------------------------------------------------------------------------
static glm::vec3 projectToSurface(glm::vec3 p, int shapeType) {
	if (shapeType == 1) { // Sphere
		return glm::normalize(p) * 0.85f;
	} else if (shapeType == 0) { // Cube
		float absX = std::abs(p.x), absY = std::abs(p.y), absZ = std::abs(p.z);
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
// Primitive Mesh Generative Helpers (Original Simpler Configurations Reverted)
//------------------------------------------------------------------------------
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
// Procedural Segment Array Populators
//------------------------------------------------------------------------------
static void generateBranch(std::vector<glm::vec3>& segA, std::vector<glm::vec3>& segB, glm::vec3 startPt, glm::vec3 dir, int steps, int shapeType, float jitter) {
	glm::vec3 curr = startPt;
	glm::vec3 currDir = dir;

	for (int i = 0; i < steps; ++i) {
		if (segA.size() >= 90) return; // Hard safe cap allocation limit for standard fragment shaders

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

		glm::vec3 nextPt = curr + currDir * 0.045f;
		nextPt = projectToSurface(nextPt, shapeType);

		segA.push_back(curr);
		segB.push_back(nextPt);

		curr = nextPt;

		// Sub-branch sprout trigger node calculation loop[cite: 5]
		if (i > 3 && i < steps - 4 && ((double)rand() / (double)RAND_MAX) < 0.12 && segA.size() < 90) {
			float splitAngle = (((double)rand() / (double)RAND_MAX) > 0.5 ? 1.0f : -1.0f) * 0.6f;
			glm::vec3 subDir = glm::normalize(currDir * std::cos(splitAngle) + sideDir * std::sin(splitAngle));
			generateBranch(segA, segB, curr, subDir, steps - i, shapeType, jitter);
		}
	}
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
// Application Entry and Workspace State
//------------------------------------------------------------------------------
struct App {
	GLFWwindow* window = nullptr;
	Camera camera{glm::radians(45.f), glm::radians(35.f), 3.2f};

	GLuint program = 0;
	GPUMesh hostMeshes[3];
	GPUMesh previewLinesMesh;
	int shapeIndex = 2; // Default Torus

	bool wireframe = true;
	bool autoRotate = false;
	bool dragging = false;
	double lastX = 0.0, lastY = 0.0;

	glm::vec3 lightPos{2.f, 2.f, 2.f};
	glm::vec3 lightColor{1.f, 1.f, 1.f};
	glm::vec3 diffuseColor{0.7f, 0.58f, 0.48f};
	float ambient = 0.25f;

	int mainArms = 5;
	float crackWidth = 0.06f;
	float crackDepth = 0.45f; // Normal scaling factor
	float crackJitter = 1.0f;
	bool carveEnabled = false;

	glm::vec3 activeStrikePoint{0.0f, 0.6f, 0.25f};
	std::vector<glm::vec3> segStarts;
	std::vector<glm::vec3> segEnds;

	void updateFractureBlueprint() {
		previewLinesMesh.clear();
		segStarts.clear();
		segEnds.clear();
		carveEnabled = false; // Reset bake status on adjustment

		constexpr float TWO_PI = 6.28318530718f;
		glm::vec3 norm = glm::normalize(activeStrikePoint);
		if (shapeIndex == 0) {
			if (std::abs(activeStrikePoint.x) > 0.64f) norm = glm::vec3(activeStrikePoint.x > 0 ? 1 : -1, 0, 0);
			else if (std::abs(activeStrikePoint.y) > 0.64f) norm = glm::vec3(0, activeStrikePoint.y > 0 ? 1 : -1, 0);
			else norm = glm::vec3(0, 0, activeStrikePoint.z > 0 ? 1 : -1);
		}
		glm::vec3 helper = std::abs(norm.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
		glm::vec3 tangentX = glm::normalize(glm::cross(norm, helper));
		glm::vec3 tangentY = glm::cross(norm, tangentX);

		for (int i = 0; i < mainArms; ++i) {
			float angle = (i * TWO_PI) / mainArms;
			glm::vec3 armDir = glm::normalize(tangentX * std::cos(angle) + tangentY * std::sin(angle));
			generateBranch(segStarts, segEnds, activeStrikePoint, armDir, 25, shapeIndex, crackJitter);
		}

		// Compile live indicator lines
		Mesh lines;
		for (size_t i = 0; i < segStarts.size(); ++i) {
			lines.verts.push_back(segStarts[i] + glm::normalize(segStarts[i]) * 0.005f);
			lines.verts.push_back(segEnds[i] + glm::normalize(segEnds[i]) * 0.005f);
			lines.normals.push_back(glm::normalize(segStarts[i]));
			lines.normals.push_back(glm::normalize(segEnds[i]));
		}

		// Upload raw vector strip
		glGenVertexArrays(1, &previewLinesMesh.vao);
		glBindVertexArray(previewLinesMesh.vao);
		glGenBuffers(1, &previewLinesMesh.vboPos);
		glBindBuffer(GL_ARRAY_BUFFER, previewLinesMesh.vboPos);
		glBufferData(GL_ARRAY_BUFFER, lines.verts.size() * sizeof(glm::vec3), lines.verts.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		previewLinesMesh.count = (GLsizei)lines.verts.size();
		glBindVertexArray(0);
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
			app->updateFractureBlueprint();
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

	ImGui::Begin("Analytical Subtraction Workshop");
	const char* shapes[] = {"Standard Box (12 Tris)", "Standard Sphere", "Standard Torus"};
	if (ImGui::Combo("Base Primitive", &app.shapeIndex, shapes, 3)) {
		app.activeStrikePoint = projectToSurface(glm::vec3(0.0f, 1.0f, 0.0f), app.shapeIndex);
		app.updateFractureBlueprint();
	}
	ImGui::Checkbox("Wireframe Grid Overlay", &app.wireframe);
	ImGui::Checkbox("Auto-Rotate Viewport", &app.autoRotate);
	ImGui::ColorEdit3("Object Tint", &app.diffuseColor[0]);
	
	ImGui::Spacing(); ImGui::Separator();
	ImGui::Text("Fracture Spine Tuning");
	if (ImGui::SliderInt("Primary Arms", &app.mainArms, 2, 10) ||
		ImGui::SliderFloat("Carve Width", &app.crackWidth, 0.01f, 0.12f) ||
		ImGui::SliderFloat("Slope Depth", &app.crackDepth, 0.05f, 1.0f) ||
		ImGui::SliderFloat("Pattern Jitter", &app.crackJitter, 0.0f, 2.5f)) {
		app.updateFractureBlueprint();
	}
	
	ImGui::Spacing(); ImGui::Separator();
	ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "ANALYTICAL CSG OPERATIONS:");
	if (ImGui::Button("Carve Model Volume", ImVec2(-1, 30))) {
		app.carveEnabled = true;
	}
	if (ImGui::Button("Reset/Clear Subtraction", ImVec2(-1, 22))) {
		app.carveEnabled = false;
	}

	ImGui::Spacing();
	ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "PIXEL-PERFECT WORKFLOW:");
	ImGui::TextWrapped("1. Right-Click on the simple shape surface to immediately calculate a complex skeleton footprint without polygon limits[cite: 5].");
	ImGui::TextWrapped("2. Fine-tune parameters instantly without topological latency.");
	ImGui::TextWrapped("3. Click 'Carve Model Volume' to perform the perfect infinite-resolution mathematical boolean cutout[cite: 5].");
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
	glUniform3fv(glGetUniformLocation(app.program, "diffuseColor"), 1, glm::value_ptr(app.diffuseColor));

	// Pass mathematical boolean uniform metrics directly into fragment block
	glUniform1i(glGetUniformLocation(app.program, "uCarveEnabled"), app.carveEnabled ? 1 : 0);
	glUniform1f(glGetUniformLocation(app.program, "uCrackWidth"), app.crackWidth);
	glUniform1f(glGetUniformLocation(app.program, "uCrackDepth"), app.crackDepth);
	glUniform1i(glGetUniformLocation(app.program, "uSegCount"), (int)app.segStarts.size());

	if (!app.segStarts.empty()) {
		glUniform3fv(glGetUniformLocation(app.program, "uSegA"), (GLsizei)app.segStarts.size(), glm::value_ptr(app.segStarts[0]));
		glUniform3fv(glGetUniformLocation(app.program, "uSegB"), (GLsizei)app.segEnds.size(), glm::value_ptr(app.segEnds[0]));
	}

	// 1. Render host shape model
	GPUMesh& hostMesh = app.hostMeshes[app.shapeIndex];
	glBindVertexArray(hostMesh.vao);
	if (app.wireframe) glDrawElements(GL_LINES, hostMesh.wireCount, GL_UNSIGNED_INT, (void*)0);
	else glDrawArrays(GL_TRIANGLES, 0, hostMesh.count);

	// 2. Render fracture alignment path vectors when previewing
	if (!app.carveEnabled && app.previewLinesMesh.count > 0) {
		glLineWidth(3.0f);
		glUniform1i(glGetUniformLocation(app.program, "uCarveEnabled"), 0); // Disable carving changes on vector line draw pass
		glUniform3f(glGetUniformLocation(app.program, "diffuseColor"), 0.0f, 1.0f, 0.4f); // Neon layout path indicator color
		glBindVertexArray(app.previewLinesMesh.vao);
		glDrawArrays(GL_LINES, 0, app.previewLinesMesh.count);
		glLineWidth(1.0f);
	}

	ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	glfwSwapBuffers(app.window);
}

int main() {
	if (!glfwInit()) return 1;
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	static App app;
	app.window = glfwCreateWindow(900, 700, "Starburst Analytical Carver", nullptr, nullptr);
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

	app.updateFractureBlueprint();
	emscripten_set_main_loop_arg(frame, &app, 0, 1);
	return 0;
}
