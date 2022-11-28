import numpy
import random
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import glm
from OBJ import *
from math import sin, cos

pygame.init()

screen = pygame.display.set_mode((1000, 800), pygame.OPENGL | pygame.DOUBLEBUF)

current_x = 0
current_y = 0
diff_x = 0
diff_y = 0
diff_z = 0
last_pos = None
angle = 0.02

vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;
uniform mat4 amatrix;
out vec3 ourColor;
out vec2 fragCoord;
void main()
{
    gl_Position = amatrix * vec4(position, 1.0f);
    fragCoord =  gl_Position.xy;
    ourColor = vertexColor;
}
"""

fragment_shader = """
#version 460
layout(location = 0) out vec4 fragColor;
in float intensity;
in float intensity2;
in vec2 vertexTexcoords;
in vec3 lPosition;
in vec4 fNormal;
in float time;
uniform sampler2D tex;
uniform vec4 diffuse;
uniform vec4 ambient;
void main()
{
	vec3 color = vec3(0.0, 0.0, 0.0);
	
	if( abs(mod( abs(lPosition.x), abs(.2 * sin(time * 10.0) )   )) < .1){
		color.x = 1.0;
	}
	
	
	if( abs(mod(lPosition.z, .2 * sin(time * 10.0 + 1.0))) < .1){
		color.z = 1.0;
	}
	gl_FragColor = vec4(color, 1.0) * intensity2;
}
"""

fragment_shader2 = """
#version 460
#define PI 3.14159265358979
layout (location = 0) out vec4 fragColor;
uniform vec3 color;
uniform float iTime;
in vec2 fragCoord;
in vec3 ourColor;
float gm(float eq, float t, float h, float s, bool i)
{
    float sg = min(abs(eq), 1.0/abs(eq)); // smooth gradient
    float og = abs(sin(eq*PI-t)); // oscillating gradient
    if (i) og = min(og, abs(sin(PI/eq+t))); // reciprocals
    return pow(1.0-og, h)*pow(sg, s);
}
void main()
{
    float t = iTime/2.0;
    float h = 5.0; // hardness
    float s = 0.25; // shadow
    bool rc = false; // reciprocals
    vec3 bg = vec3(0); // black background
    vec2 R = vec2(1, 1);
    
    float aa = 3.0; // anti-aliasing
    for (float j = 0.0; j < aa; j++)
    for (float k = 0.0; k < aa; k++)
    {
        vec3 c = vec3(0);
        vec2 o = vec2(j, k)/aa;
        vec2 sc = (fragCoord-0.5*R+o)/R.y; // screen coords
        float x2 = sc.x*sc.x;
        float y2 = sc.y*sc.y;
        // square root grids
        c += gm(x2, t, h, s, rc); // x
        c += gm(y2, 0.0, h, s, rc); // y
        c += gm(x2+y2, t, h, s, rc); // addition
        c += gm(x2-y2, t, h, s, rc); // subtraction
        c += gm(x2*y2, t, h, s, rc); // multiplication
        c += gm(x2/y2, t, h, s, rc); // division
        
        bg += c;
    }
    bg /= aa*aa;
    
    bg *= sqrt(bg)*0.5; // brightness & contrast
    fragColor = vec4(bg, 1.0);
}
"""

fragment_shader3 = """
#version 460
#define NUM_LAYER 8.
#define PI 3.14159265358979
layout (location = 0) out vec4 fragColor;
uniform vec3 color;
uniform float iTime;
in vec2 fragCoord;
in vec3 ourColor;
mat2 Rot(float angle){
    float s=sin(angle), c=cos(angle);
    return mat2(c, -s, s, c);
}
//random number between 0 and 1
float Hash21(vec2 p){
    p = fract(p*vec2(123.34, 456.21));
    p +=dot(p, p+45.32);
    return  fract(p.x*p.y);
}
float Star(vec2 uv, float flare){
    float d = length(uv);//center of screen is origin of uv -- length give us distance from every pixel to te center
    float m = .05/d;
    float rays = max(0., 1.-abs(uv.x*uv.y*1000.));
    m +=rays*flare;
    
    uv *=Rot(3.1415/4.);
    rays = max(0., 1.-abs(uv.x*uv.y*1000.));
    m +=rays*.3*flare;
    m *=smoothstep(1., .2, d);
    return m;
}
vec3 StarLayer(vec2 uv){
   
   vec3 col = vec3(0.);
   
    vec2 gv= fract(uv)-.5; //gv is grid view
    vec2 id= floor(uv);
    
    for(int y=-1; y<=1; y++){
        for(int x=-1; x<=1; x++){
            
            vec2 offset= vec2(x, y);
            float n = Hash21(id+offset);
            float size = fract(n*345.32);
                float star= Star(gv-offset-(vec2(n, fract(n*34.))-.5), smoothstep(.9, 1., size)*.6);
            vec3 color = sin(vec3(.2, .3, .9)*fract(n*2345.2)*123.2)*.5+.5;
            color = color*vec3(1., .25, 1.+size);
            
            star *=sin(iTime*3.+n*6.2831)*.5+1.;
            col +=star*size*color; 
            
         }
     }
    return col;
}
void main()
{
    vec2 iResolution = vec2(10, 10);
    vec2 iMouse = vec2(10, 10);
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    float t=  iTime*.02;
    vec2 M = (iMouse.xy-iResolution.xy*.5)/iResolution.y;
    uv *=Rot(t);
    uv +=M*4.;
    
    vec3 col = vec3(0.);
    
    for(float i =0.; i<1.; i += 1./NUM_LAYER){
        float depth = fract(i+t);
        float scale= mix(10.,.5, depth);
        float fade = depth*smoothstep(1., .9, depth);
        col += StarLayer(uv*scale+i*453.32-M)*fade;
    }
    fragColor = vec4(col,1.0);
}
"""

fragment_shader4 = """
#version 460
#define PI 3.14159265358979
layout (location = 0) out vec4 fragColor;
uniform vec3 color;
uniform float iTime;
in vec2 fragCoord;
in vec3 ourColor;
void main()
{
    vec2 iResolution = vec2(10, 10);
    fragColor = 9./max((fragCoord-iResolution.xy*.5)*mat2(cos(iTime-log(length(fragCoord))+vec4(0,11,33,0)))+9.,.1).xyyy;
}
"""

compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)

compiled_fragment_shader = compileShader(fragment_shader2, GL_FRAGMENT_SHADER)
compiled_fragment_shader2 = compileShader(fragment_shader2, GL_FRAGMENT_SHADER)
compiled_fragment_shader3 = compileShader(fragment_shader3, GL_FRAGMENT_SHADER)
compiled_fragment_shader4 = compileShader(fragment_shader4, GL_FRAGMENT_SHADER)

shader = compileProgram(compiled_vertex_shader, compiled_fragment_shader)

shader2 = compileProgram(compiled_vertex_shader, compiled_fragment_shader2)

shader3 = compileProgram(compiled_vertex_shader, compiled_fragment_shader3)

shader4 = compileProgram(compiled_vertex_shader, compiled_fragment_shader4)

glUseProgram(shader2)
glEnable(GL_DEPTH_TEST)

obj = Obj("cocacola.obj")

vertices = []
texcoords = []
normals = []
for face in obj.faces:
    for v in range(len(face)):
        vertices.append((obj.vertices[face[v][0] - 1]))
        texcoords.append((obj.texcoords[face[v][1] - 1]))
        normals.append((obj.normals[face[v][2] - 1]))

vertex_data = numpy.hstack(
    [
        numpy.array(vertices, dtype=numpy.float32),
        numpy.array(texcoords, dtype=numpy.float32),
        numpy.array(normals, dtype=numpy.float32),
    ]
)

faces = []
for face in obj.faces:
    for f in face:
        faces.append(int(f[0]) - 1)

index_data = numpy.hstack(
    [
        numpy.array(faces, dtype=numpy.int32),
    ]
)

index_data = numpy.array(faces, dtype=numpy.int32)

vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

element_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)


def calculateMatrix(angle):
    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate = glm.rotate(i, glm.radians(angle), glm.vec3(0, 1, 0))
    scale = glm.scale(i, glm.vec3(1, 1, 1))

    model = translate * rotate * scale

    view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    projection = glm.perspective(glm.radians(45), 1600 / 1200, 0.1, 1000.0)

    amatrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader, "amatrix"), 1, GL_FALSE, glm.value_ptr(amatrix)
    )


glViewport(0, 0, 1000, 800)


running = True

glClearColor(0, 0, 0, 1.0)

r = 0
a = 0
x = 0
y = 0

current_shader = shader
while running:
    r += 1

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(current_shader)

    glUniform1f(glGetUniformLocation(shader, "iTime"), r / 100)

    glDrawElements(GL_TRIANGLES, len(index_data), GL_UNSIGNED_INT, None)

    calculateMatrix(a)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                x -= 10
            if event.key == pygame.K_d:
                x += 10
            if event.key == pygame.K_w:
                y += 10
            if event.key == pygame.K_s:
                y -= 10
            if event.key == pygame.K_1:
                current_shader = shader
            if event.key == pygame.K_2:
                current_shader = shader3
            if event.key == pygame.K_3:
                current_shader = shader4
