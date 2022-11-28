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
    ourColor = vertexColor;
    fragCoord = gl_Position.xy;
}
"""

# Referencia https://www.shadertoy.com/view/4tc3WB
fragment_shader = """
#version 460

vec2 iResolution = vec2(2, 2);
vec2 iMouse = vec2(10, 10);
vec2 iChannel0 = vec2(10, 10);

layout (location = 0) out vec4 fragColor;
in vec2 fragCoord;
uniform float iTime;
float gTime;

// --------------------------------------------------------
// OPTIONS
// --------------------------------------------------------

// Disable to see more colour variety
#define SEAMLESS_LOOP
//#define COLOUR_CYCLE

#define PI 3.14159265359
#define PHI (1.618033988749895)

float t;

#define saturate(x) clamp(x, 0., 1.)


// --------------------------------------------------------
// http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
// --------------------------------------------------------

mat3 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
    );
}


// --------------------------------------------------------
// http://math.stackexchange.com/a/897677
// --------------------------------------------------------

mat3 orientMatrix(vec3 A, vec3 B) {
    mat3 Fi = mat3(
        A,
        (B - dot(A, B) * A) / length(B - dot(A, B) * A),
        cross(B, A)
    );
    mat3 G = mat3(
        dot(A, B),              -length(cross(A, B)),   0,
        length(cross(A, B)),    dot(A, B),              0,
        0,                      0,                      1
    );
    return Fi * G * inverse(Fi);
}


// --------------------------------------------------------
// HG_SDF
// https://www.shadertoy.com/view/Xs3GRB
// --------------------------------------------------------

#define GDFVector3 normalize(vec3(1, 1, 1 ))
#define GDFVector3b normalize(vec3(-1, -1, -1 ))
#define GDFVector4 normalize(vec3(-1, 1, 1))
#define GDFVector4b normalize(vec3(-1, -1, 1))
#define GDFVector5 normalize(vec3(1, -1, 1))
#define GDFVector5b normalize(vec3(1, -1, -1))
#define GDFVector6 normalize(vec3(1, 1, -1))
#define GDFVector6b normalize(vec3(-1, 1, -1))

#define GDFVector7 normalize(vec3(0, 1, PHI+1.))
#define GDFVector7b normalize(vec3(0, 1, -PHI-1.))
#define GDFVector8 normalize(vec3(0, -1, PHI+1.))
#define GDFVector8b normalize(vec3(0, -1, -PHI-1.))
#define GDFVector9 normalize(vec3(PHI+1., 0, 1))
#define GDFVector9b normalize(vec3(PHI+1., 0, -1))
#define GDFVector10 normalize(vec3(-PHI-1., 0, 1))
#define GDFVector10b normalize(vec3(-PHI-1., 0, -1))
#define GDFVector11 normalize(vec3(1, PHI+1., 0))
#define GDFVector11b normalize(vec3(1, -PHI-1., 0))
#define GDFVector12 normalize(vec3(-1, PHI+1., 0))
#define GDFVector12b normalize(vec3(-1, -PHI-1., 0))

#define GDFVector13 normalize(vec3(0, PHI, 1))
#define GDFVector13b normalize(vec3(0, PHI, -1))
#define GDFVector14 normalize(vec3(0, -PHI, 1))
#define GDFVector14b normalize(vec3(0, -PHI, -1))
#define GDFVector15 normalize(vec3(1, 0, PHI))
#define GDFVector15b normalize(vec3(1, 0, -PHI))
#define GDFVector16 normalize(vec3(-1, 0, PHI))
#define GDFVector16b normalize(vec3(-1, 0, -PHI))
#define GDFVector17 normalize(vec3(PHI, 1, 0))
#define GDFVector17b normalize(vec3(PHI, -1, 0))
#define GDFVector18 normalize(vec3(-PHI, 1, 0))
#define GDFVector18b normalize(vec3(-PHI, -1, 0))

#define fGDFBegin float d = 0.;

// Version with variable exponent.
// This is slow and does not produce correct distances, but allows for bulging of objects.
#define fGDFExp(v) d += pow(abs(dot(p, v)), e);

// Version with without exponent, creates objects with sharp edges and flat faces
#define fGDF(v) d = max(d, abs(dot(p, v)));

#define fGDFExpEnd return pow(d, 1./e) - r;
#define fGDFEnd return d - r;

// Primitives follow:

float fDodecahedron(vec3 p, float r) {
    fGDFBegin
    fGDF(GDFVector13) fGDF(GDFVector14) fGDF(GDFVector15) fGDF(GDFVector16)
    fGDF(GDFVector17) fGDF(GDFVector18)
    fGDFEnd
}

float fIcosahedron(vec3 p, float r) {
    fGDFBegin
    fGDF(GDFVector3) fGDF(GDFVector4) fGDF(GDFVector5) fGDF(GDFVector6)
    fGDF(GDFVector7) fGDF(GDFVector8) fGDF(GDFVector9) fGDF(GDFVector10)
    fGDF(GDFVector11) fGDF(GDFVector12)
    fGDFEnd
}

float vmax(vec3 v) {
    return max(max(v.x, v.y), v.z);
}

float sgn(float x) {
	return (x<0.)?-1.:1.;
}

// Plane with normal n (n is normalized) at some distance from the origin
float fPlane(vec3 p, vec3 n, float distanceFromOrigin) {
    return dot(p, n) + distanceFromOrigin;
}

// Box: correct distance to corners
float fBox(vec3 p, vec3 b) {
	vec3 d = abs(p) - b;
	return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}

// Distance to line segment between <a> and <b>, used for fCapsule() version 2below
float fLineSegment(vec3 p, vec3 a, vec3 b) {
	vec3 ab = b - a;
	float t = saturate(dot(p - a, ab) / dot(ab, ab));
	return length((ab*t + a) - p);
}

// Capsule version 2: between two end points <a> and <b> with radius r 
float fCapsule(vec3 p, vec3 a, vec3 b, float r) {
	return fLineSegment(p, a, b) - r;
}

// Rotate around a coordinate axis (i.e. in a plane perpendicular to that axis) by angle <a>.
// Read like this: R(p.xz, a) rotates "x towards z".
// This is fast if <a> is a compile-time constant and slower (but still practical) if not.
void pR(inout vec2 p, float a) {
    p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Reflect space at a plane
float pReflect(inout vec3 p, vec3 planeNormal, float offset) {
    float t = dot(p, planeNormal)+offset;
    if (t < 0.) {
        p = p - (2.*t)*planeNormal;
    }
    return sign(t);
}

// Repeat around the origin by a fixed angle.
// For easier use, num of repetitions is use to specify the angle.
float pModPolar(inout vec2 p, float repetitions) {
	float angle = 2.*PI/repetitions;
	float a = atan(p.y, p.x) + angle/2.;
	float r = length(p);
	float c = floor(a/angle);
	a = mod(a,angle) - angle/2.;
	p = vec2(cos(a), sin(a))*r;
	// For an odd number of repetitions, fix cell index of the cell in -x direction
	// (cell index would be e.g. -5 and 5 in the two halves of the cell):
	if (abs(c) >= (repetitions/2.)) c = abs(c);
	return c;
}

// Repeat around an axis
void pModPolar(inout vec3 p, vec3 axis, float repetitions, float offset) {
    vec3 z = vec3(0,0,1);
	mat3 m = orientMatrix(axis, z);
    p *= inverse(m);
    pR(p.xy, offset);
    pModPolar(p.xy, repetitions);
    pR(p.xy, -offset);
    p *= m;
}


// --------------------------------------------------------
// knighty
// https://www.shadertoy.com/view/MsKGzw
// --------------------------------------------------------

int Type=5;
vec3 nc;
vec3 pbc;
vec3 pca;
void initIcosahedron() {//setup folding planes and vertex
    float cospin=cos(PI/float(Type)), scospin=sqrt(0.75-cospin*cospin);
    nc=vec3(-0.5,-cospin,scospin);//3rd folding plane. The two others are xz and yz planes
	pbc=vec3(scospin,0.,0.5);//No normalization in order to have 'barycentric' coordinates work evenly
	pca=vec3(0.,scospin,cospin);
	pbc=normalize(pbc);	pca=normalize(pca);//for slightly better DE. In reality it's not necesary to apply normalization :) 

}

void pModIcosahedron(inout vec3 p) {
    p = abs(p);
    pReflect(p, nc, 0.);
    p.xy = abs(p.xy);
    pReflect(p, nc, 0.);
    p.xy = abs(p.xy);
    pReflect(p, nc, 0.);
}

float indexSgn(float s) {
	return s / 2. + 0.5;
}

bool boolSgn(float s) {
	return bool(s / 2. + 0.5);
}

float pModIcosahedronIndexed(inout vec3 p, int subdivisions) {
	float x = indexSgn(sgn(p.x));
	float y = indexSgn(sgn(p.y));
	float z = indexSgn(sgn(p.z));
    p = abs(p);
	pReflect(p, nc, 0.);

	float xai = sgn(p.x);
	float yai = sgn(p.y);
    p.xy = abs(p.xy);
	float sideBB = pReflect(p, nc, 0.);

	float ybi = sgn(p.y);
	float xbi = sgn(p.x);
    p.xy = abs(p.xy);
	pReflect(p, nc, 0.);
    
    float idx = 0.;

    float faceGroupAi = indexSgn(ybi * yai * -1.);
    float faceGroupBi = indexSgn(yai);
    float faceGroupCi = clamp((xai - ybi -1.), 0., 1.);
    float faceGroupDi = clamp(1. - faceGroupAi - faceGroupBi - faceGroupCi, 0., 1.);

    idx += faceGroupAi * (x + (2. * y) + (4. * z));
    idx += faceGroupBi * (8. + y + (2. * z));
    # ifndef SEAMLESS_LOOP
    	idx += faceGroupCi * (12. + x + (2. * z));
    # endif
    idx += faceGroupDi * (12. + x + (2. * y));

	return idx;
}


// --------------------------------------------------------
// IQ
// https://www.shadertoy.com/view/ll2GD3
// --------------------------------------------------------

vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
    return a + b*cos( 6.28318*(c*t+d) );
}

vec3 spectrum(float n) {
    return pal( n, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}


// --------------------------------------------------------
// tdhooper
// https://www.shadertoy.com/view/Mtc3RX
// --------------------------------------------------------

vec3 vMin(vec3 p, vec3 a, vec3 b, vec3 c) {
    float la = length(p - a);
    float lb = length(p - b);
    float lc = length(p - c);
    if (la < lb) {
        if (la < lc) {
            return a;
        } else {
            return c;
        }
    } else {
        if (lb < lc) {
            return b;
        } else {
            return c;
        }
    }
}

// Nearest icosahedron vertex
vec3 icosahedronVertex(vec3 p) {
    if (p.z > 0.) {
        if (p.x > 0.) {
            if (p.y > 0.) {
                return vMin(p, GDFVector13, GDFVector15, GDFVector17);
            } else {
                return vMin(p, GDFVector14, GDFVector15, GDFVector17b);
            }
        } else {
            if (p.y > 0.) {
                return vMin(p, GDFVector13, GDFVector16, GDFVector18);
            } else {
                return vMin(p, GDFVector14, GDFVector16, GDFVector18b);
            }
        }
    } else {
        if (p.x > 0.) {
            if (p.y > 0.) {
                return vMin(p, GDFVector13b, GDFVector15b, GDFVector17);
            } else {
                return vMin(p, GDFVector14b, GDFVector15b, GDFVector17b);
            }
        } else {
            if (p.y > 0.) {
                return vMin(p, GDFVector13b, GDFVector16b, GDFVector18);
            } else {
                return vMin(p, GDFVector14b, GDFVector16b, GDFVector18b);
            }
        }
    }
}

// Nearest vertex and distance.
// Distance is roughly to the boundry between the nearest and next
// nearest icosahedron vertices, ensuring there is always a smooth
// join at the edges, and normalised from 0 to 1
vec4 icosahedronAxisDistance(vec3 p) {
    vec3 iv = icosahedronVertex(p);
    vec3 originalIv = iv;

    vec3 pn = normalize(p);
    pModIcosahedron(pn);
    pModIcosahedron(iv);

    float boundryDist = dot(pn, vec3(1, 0, 0));
    float boundryMax = dot(iv, vec3(1, 0, 0));
    boundryDist /= boundryMax;

    float roundDist = length(iv - pn);
    float roundMax = length(iv - vec3(0, 0, 1.));
    roundDist /= roundMax;
    roundDist = -roundDist + 1.;

    float blend = 1. - boundryDist;
	blend = pow(blend, 6.);
    
    float dist = mix(roundDist, boundryDist, blend);

    return vec4(originalIv, dist);
}

// Twists p around the nearest icosahedron vertex
void pTwistIcosahedron(inout vec3 p, float amount) {
    vec4 a = icosahedronAxisDistance(p);
    vec3 axis = a.xyz;
    float dist = a.a;
    mat3 m = rotationMatrix(axis, dist * amount);
    p *= m;
}


// --------------------------------------------------------
// MAIN
// --------------------------------------------------------

struct Model {
    float dist;
    vec3 colour;
    float id;
};
     
Model fInflatedIcosahedron(vec3 p, vec3 axis) {
    float d = 1000.;
    
    # ifdef SEAMLESS_LOOP
    	// Radially repeat along the rotation axis, so the
    	// colours repeat more frequently and we can use
    	// less frames for a seamless loop
    	pModPolar(p, axis, 3., PI/2.);
	# endif
    
    // Slightly inflated icosahedron
    float idx = pModIcosahedronIndexed(p, 0);
    d = min(d, dot(p, pca) - .9);
    d = mix(d, length(p) - .9, .5);

    // Colour each icosahedron face differently
    # ifdef SEAMLESS_LOOP
    	if (idx == 3.) {
    		idx = 2.;
    	}
    	idx /= 10.;
   	# else
    	idx /= 20.;
    # endif
    # ifdef COLOUR_CYCLE
    	idx = mod(idx + t*1.75, 1.);
    # endif
    vec3 colour = spectrum(idx);
    
    d *= .6;
	return Model(d, colour, 1.);
}

void pTwistIcosahedron(inout vec3 p, vec3 center, float amount) {
    p += center;
    pTwistIcosahedron(p, 5.5);
    p -= center;
}

Model model(vec3 p) {
    float rate = PI/6.;
    vec3 axis = pca;

    vec3 twistCenter = vec3(0);
    twistCenter.x = cos(t * rate * -3.) * .3;
	twistCenter.y = sin(t * rate * -3.) * .3;

	mat3 m = rotationMatrix(
        reflect(axis, vec3(0,1,0)),
        t * -rate
   	);
    p *= m;
    twistCenter *= m;

    pTwistIcosahedron(p, twistCenter, 5.5);

	return fInflatedIcosahedron(p, axis);
}


// The MINIMIZED version of https://www.shadertoy.com/view/Xl2XWt


const float MAX_TRACE_DISTANCE = 30.0;           // max trace distance
const float INTERSECTION_PRECISION = 0.001;        // precision of the intersection
const int NUM_OF_TRACE_STEPS = 100;


// checks to see which intersection is closer
// and makes the y of the vec2 be the proper id
vec2 opU( vec2 d1, vec2 d2 ){
    return (d1.x<d2.x) ? d1 : d2;
}

//--------------------------------
// Modelling
//--------------------------------
Model map( vec3 p ){
    return model(p);
}

// LIGHTING

float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
        float h = map( ro + rd*t ).dist;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}


float calcAO( in vec3 pos, in vec3 nor )
{
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).dist;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

const float GAMMA = 2.2;

vec3 gamma(vec3 color, float g)
{
    return pow(color, vec3(g));
}

vec3 linearToScreen(vec3 linearRGB)
{
    return gamma(linearRGB, 1.0 / GAMMA);
}

vec3 doLighting(vec3 col, vec3 pos, vec3 nor, vec3 ref, vec3 rd) {

    // lighitng        
    float occ = calcAO( pos, nor );
    vec3  lig = normalize( vec3(-0.6, 0.7, 0.5) );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    //float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );
    //float spe = pow(clamp( dot( ref, lig ), 0.0, 1.0 ),16.0);
    
    dif *= softshadow( pos, lig, 0.02, 2.5 );
    //dom *= softshadow( pos, ref, 0.02, 2.5 );

    vec3 lin = vec3(0.0);
    lin += 1.20*dif*vec3(.95,0.80,0.60);
    //lin += 1.20*spe*vec3(1.00,0.85,0.55)*dif;
    lin += 0.80*amb*vec3(0.50,0.70,.80)*occ;
    //lin += 0.30*dom*vec3(0.50,0.70,1.00)*occ;
    lin += 0.30*bac*vec3(0.25,0.25,0.25)*occ;
    lin += 0.20*fre*vec3(1.00,1.00,1.00)*occ;
    col = col*lin;

    return col;
}

struct Hit {
    float len;
    vec3 colour;
    float id;
};

Hit calcIntersection( in vec3 ro, in vec3 rd ){

    float h =  INTERSECTION_PRECISION*2.0;
    float t = 0.0;
    float res = -1.0;
    float id = -1.;
    vec3 colour;

    for( int i=0; i< NUM_OF_TRACE_STEPS ; i++ ){

        if( abs(h) < INTERSECTION_PRECISION || t > MAX_TRACE_DISTANCE ) break;
        Model m = map( ro+rd*t );
        h = m.dist;
        t += h;
        id = m.id;
        colour = m.colour;
    }

    if( t < MAX_TRACE_DISTANCE ) res = t;
    if( t > MAX_TRACE_DISTANCE ) id =-1.0;

    return Hit( res , colour , id );
}


//----
// Camera Stuffs
//----
mat3 calcLookAtMatrix( in vec3 ro, in vec3 ta, in float roll )
{
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(sin(roll),cos(roll),0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    return mat3( uu, vv, ww );
}

void doCamera(out vec3 camPos, out vec3 camTar, out float camRoll, in float time, in vec2 mouse) {

    float x = mouse.x;
    float y = mouse.y;
    
    x = .65;
    y = .44;
    
    float dist = 3.3;
    float height = 0.;
    camPos = vec3(0,0,-dist);
    vec3 axisY = vec3(0,1,0);
    vec3 axisX = vec3(1,0,0);
    mat3 m = rotationMatrix(axisY, -x * PI * 2.);
    axisX *= m;
    camPos *= m;
    m = rotationMatrix(axisX, -(y -.5) * PI*2.);
    camPos *= m;
    camPos.y += height;
    camTar = -camPos + vec3(.0001);
    camTar.y += height;
    camRoll = 0.;
}

// Calculates the normal by taking a very small distance,
// remapping the function, and getting normal for that
vec3 calcNormal( in vec3 pos ){

    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
        map(pos+eps.xyy).dist - map(pos-eps.xyy).dist,
        map(pos+eps.yxy).dist - map(pos-eps.yxy).dist,
        map(pos+eps.yyx).dist - map(pos-eps.yyx).dist );
    return normalize(nor);
}

vec2 ffragCoord;

vec3 render( Hit hit , vec3 ro , vec3 rd ){

    vec3 pos = ro + rd * hit.len;

    vec3 color = vec3(.04,.045,.05);
    color = vec3(.35, .5, .65);
    vec3 colorB = vec3(.8, .8, .9);
    
    vec2 pp = (-iResolution.xy + 2.0*ffragCoord.xy)/iResolution.y;
    
    color = mix(colorB, color, length(pp)/1.5);


    if (hit.id == 1.){
        vec3 norm = calcNormal( pos );
        vec3 ref = reflect(rd, norm);
        color = doLighting(hit.colour, pos, norm, ref, rd);
    }

  return color;
}


void main()
{
    initIcosahedron();
    t = iTime - .25;
    //t = mod(t, 4.);
    
    ffragCoord = fragCoord;

    vec2 p = (-iResolution.xy + 2.0*fragCoord.xy)/iResolution.y;
    vec2 m = iMouse.xy / iResolution.xy;

    vec3 camPos = vec3( 0., 0., 2.);
    vec3 camTar = vec3( 0. , 0. , 0. );
    float camRoll = 0.;

    // camera movement
    doCamera(camPos, camTar, camRoll, iTime, m);

    // camera matrix
    mat3 camMat = calcLookAtMatrix( camPos, camTar, camRoll );  // 0.0 is the camera roll

    // create view ray
    vec3 rd = normalize( camMat * vec3(p.xy,2.0) ); // 2.0 is the lens length

    Hit hit = calcIntersection( camPos , rd  );


    vec3 color = render( hit , camPos , rd );
	color = linearToScreen(color);
    
    fragColor = vec4(color,1.0);
}

"""

# Referencia https://www.shadertoy.com/view/llcXW7
fragment_shader2 = """
#version 460

vec2 iResolution = vec2(2, 2);
layout (location = 0) out vec4 fragColor;
in vec2 fragCoord;
uniform float iTime;
float gTime;

#define TAU 6.28318530718

#define TILING_FACTOR 1.0
#define MAX_ITER 8


float waterHighlight(vec2 p, float time, float foaminess)
{
    vec2 i = vec2(p);
	float c = 0.0;
    float foaminess_factor = mix(1.0, 6.0, foaminess);
	float inten = .005 * foaminess_factor;

	for (int n = 0; n < MAX_ITER; n++) 
	{
		float t = time * (1.0 - (3.5 / float(n+1)));
		i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
		c += 1.0/length(vec2(p.x / (sin(i.x+t)),p.y / (cos(i.y+t))));
	}
	c = 0.2 + c / (inten * float(MAX_ITER));
	c = 1.17-pow(c, 1.4);
    c = pow(abs(c), 8.0);
	return c / sqrt(foaminess_factor);
}


void main() 
{
	float time = iTime * 0.1+23.0;
	vec2 uv = fragCoord.xy / iResolution.xy;
	vec2 uv_square = vec2(uv.x * iResolution.x / iResolution.y, uv.y);
    float dist_center = pow(2.0*length(uv - 0.5), 2.0);
    
    float foaminess = smoothstep(0.4, 1.8, dist_center);
    float clearness = 0.1 + 0.9*smoothstep(0.1, 0.5, dist_center);
    
	vec2 p = mod(uv_square*TAU*TILING_FACTOR, TAU)-250.0;
    
    float c = waterHighlight(p, time, foaminess);
    
    vec3 water_color = vec3(0.0, 0.35, 0.5);
	vec3 color = vec3(c);
    color = clamp(color + water_color, 0.0, 1.0);
    
    color = mix(water_color, color, clearness);

	fragColor = vec4(color, 1.0);
}
"""

# Referencia https://www.shadertoy.com/view/ldB3Dt
fragment_shader3 = """
#version 460

vec2 iResolution = vec2(2, 2);
layout (location = 0) out vec4 fragColor;
in vec2 fragCoord;
uniform float iTime;
float gTime;

#define FARCLIP    35.0

#define MARCHSTEPS 60
#define AOSTEPS    8
#define SHSTEPS    10
#define SHPOWER    3.0

#define PI         3.14
#define PI2        PI*0.5    

#define AMBCOL     vec3(1.0,1.0,1.0)
#define BACCOL     vec3(1.0,1.0,1.0)
#define DIFCOL     vec3(1.0,1.0,1.0)

#define MAT1       1.0

#define FOV 1.0


/***********************************************/
float rbox(vec3 p, vec3 s, float r) {	
    return length(max(abs(p)-s+vec3(r),0.0))-r;
}
float torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}
float cylinder(vec3 p, vec2 h) {
    return max( length(p.xz)-h.x, abs(p.y)-h.y );
}

/***********************************************/
void oprep2(inout vec2 p, float l, float s, float k) {
	float r=1./l;
	float ofs=s+s/(r*2.0);
	float a= mod( atan(p.x, p.y) + PI2*r*k, PI*r) -PI2*r;
	p.xy=vec2(sin(a),cos(a))*length(p.xy) -ofs;
	p.x+=ofs;
}

float hash(float n) { 
	return fract(sin(n)*43758.5453123); 
}

float noise3(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + p.z*113.0;
    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

float sminp(float a, float b) {
    const float k=0.1;
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}


/***********************************************/

vec2 DE(vec3 p) {
    
    //distortion
    float d3=noise3(p*2.0 + iTime)*0.18;
    //shape
    float h=torus(p, vec2(3.0,1.5)) -d3;
    float h2=torus(p, vec2(3.0,1.45)) -d3;
        vec3 q=p.yzx; p.yz=q.yx;
        oprep2(p.xy,32.0,0.15, 0.0);
        oprep2(p.yz,14.0,0.15, 0.0);
        float flag=p.z;
        float k=rbox(p,vec3(0.05,0.05,1.0),0.0) ;
        if (flag>0.1) k-=flag*0.18; else k-=0.01 ;

    //pipes
    p=q.zyx;

    oprep2(p.xy,3.0,8.5, 3.0);
    oprep2(p.xz,12.0,0.25, 0.0);
        
    p.y=mod(p.y,0.3)-0.5*0.3;
    float k2=rbox(p,vec3(0.12,0.12,1.0),0.05) - 0.01;

    p=q.xzy;
    float r=p.y*0.02+sin(iTime)*0.05;
        oprep2(p.zy,3.0,8.5, 0.0);
    float g=cylinder(p,vec2(1.15+r,17.0)) - sin(p.y*1.3 - iTime*4.0)*0.1 -d3;
    float g2=cylinder(p,vec2(1.05+r,18.0)) - sin(p.y*1.3 - iTime*4.0)*0.1 -d3;

      float tot=max(h,-h2);
      float sub=max(g,-g2);
        float o=max(tot,-g);
        float i=max(sub,-h);
        
            o=max(o,-k);
            i=max(i,-k2);
      
      tot=sminp(o,i);

	return vec2( tot*0.9 , MAT1);
}
/***********************************************/
vec3 normal(vec3 p) {
	vec3 e=vec3(0.01,-0.01,0.0);
	return normalize( vec3(	e.xyy*DE(p+e.xyy).x +	e.yyx*DE(p+e.yyx).x +	e.yxy*DE(p+e.yxy).x +	e.xxx*DE(p+e.xxx).x));
}
/***********************************************/
float calcAO(vec3 p, vec3 n ){
	float ao = 0.0;
	float sca = 1.0;
	for (int i=0; i<AOSTEPS; i++) {
        	float h = 0.01 + 1.2*pow(float(i)/float(AOSTEPS),1.5);
        	float dd = DE( p+n*h ).x;
        	ao += -(dd-h)*sca;
        	sca *= 0.65;
    	}
   return clamp( 1.0 - 1.0*ao, 0.0, 1.0 );
 //  return clamp(ao,0.0,1.0);
}
/***********************************************/
float calcSh( vec3 ro, vec3 rd, float s, float e, float k ) {
	float res = 1.0;
    for( int i=0; i<SHSTEPS; i++ ) {
    	if( s>e ) break;
        float h = DE( ro + rd*s ).x;
        res = min( res, k*h/s );
    	s += 0.02*SHPOWER;
    }
    return clamp( res, 0.0, 1.0 );
}
/***********************************************/
void rot( inout vec3 p, vec3 r) {
	float sa=sin(r.y); float sb=sin(r.x); float sc=sin(r.z);
	float ca=cos(r.y); float cb=cos(r.x); float cc=cos(r.z);
	p*=mat3( cb*cc, cc*sa*sb-ca*sc, ca*cc*sb+sa*sc,	cb*sc, ca*cc+sa*sb*sc, -cc*sa+ca*sb*sc,	-sb, cb*sa, ca*cb );
}
/***********************************************/
void main() {
    vec2 p = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;
    p.x *= iResolution.x/iResolution.y;	
	vec3 ta = vec3(0.0, 0.0, 0.0);
	vec3 ro =vec3(0.0, 0.0, -15.0);
	vec3 lig=normalize(vec3(2.3, 3.0, 0.0));
	
//	vec2 mp=iMouse.xy/iResolution.xy;
//	rot(ro,vec3(mp.x,mp.y,0.0));
//	rot(lig,vec3(mp.x,mp.y,0.0));
	
    float a=iTime*0.5;
    float b=sin(iTime*0.25)*0.75;
	rot(ro,vec3(a,b,0.0));
	rot(lig,vec3(a,b,0.0));

	vec3 cf = normalize( ta - ro );
    vec3 cr = normalize( cross(cf,vec3(0.0,1.0,0.0) ) );
    vec3 cu = normalize( cross(cr,cf));
	vec3 rd = normalize( p.x*cr + p.y*cu + 2.5*cf );

	vec3 col=vec3(0.0);
	/* trace */
	vec2 r=vec2(0.0);	
	float d=0.0;
	vec3 ww;
	for(int i=0; i<MARCHSTEPS; i++) {
		ww=ro+rd*d;
		r=DE(ww);		
        if( abs(r.x)<0.00 || r.x>FARCLIP ) break;
        d+=r.x;
	}
    r.x=d;
	/* draw */
	if( r.x<FARCLIP ) {
	    vec2 rs=vec2(0.2,1.0);  //rim and spec
		if (r.y==MAT1) { col=vec3(0.29,0.53,0.91);  } 

		vec3 nor=normal(ww);

    	float amb= 1.0;		
    	float dif= clamp(dot(nor, lig), 0.0,1.0);
    	float bac= clamp(dot(nor,-lig), 0.0,1.0);
    	float rim= pow(1.+dot(nor,rd), 3.0);
    	float spe= pow(clamp( dot( lig, reflect(rd,nor) ), 0.5, 1.0 ) ,16.0 );
    	float ao= calcAO(ww, nor);
    	float sh= calcSh(ww, lig, 0.01, 2.0, 4.0);

	    col *= 0.5*amb*AMBCOL*ao + 0.4*dif*DIFCOL*sh + 0.05*bac*BACCOL*ao;
	    col += 0.3*rim*amb * rs.x;
    	col += 0.5*pow(spe,1.0)*sh * rs.y;
        
	}
	
	col*=exp(.08*-r.x); col*=2.0;
	
	fragColor = vec4( col, 1.0 );
}

"""

# Refrencia https://www.shadertoy.com/view/MlS3Rh
fragment_shader4 = """
#version 460

vec2 iResolution = vec2(2, 2);
layout (location = 0) out vec4 fragColor;
in vec2 fragCoord;
uniform float iTime;
float gTime;

// "Vortex Street" by dr2 - 2015
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

// Motivated by implementation of van Wijk's IBFV by eiffie (lllGDl) and andregc (4llGWl) 

const vec4 cHashA4 = vec4 (0., 1., 57., 58.);
const vec3 cHashA3 = vec3 (1., 57., 113.);
const float cHashM = 43758.54;

vec4 Hashv4f (float p)
{
  return fract (sin (p + cHashA4) * cHashM);
}

float Noisefv2 (vec2 p)
{
  vec2 i = floor (p);
  vec2 f = fract (p);
  f = f * f * (3. - 2. * f);
  vec4 t = Hashv4f (dot (i, cHashA3.xy));
  return mix (mix (t.x, t.y, f.x), mix (t.z, t.w, f.x), f.y);
}

float Fbm2 (vec2 p)
{
  float s = 0.;
  float a = 1.;
  for (int i = 0; i < 6; i ++) {
    s += a * Noisefv2 (p);
    a *= 0.5;
    p *= 2.;
  }
  return s;
}

float tCur;

vec2 VortF (vec2 q, vec2 c)
{
  vec2 d = q - c;
  return 0.25 * vec2 (d.y, - d.x) / (dot (d, d) + 0.05);
}

vec2 FlowField (vec2 q)
{
  vec2 vr, c;
  float dir = 1.;
  c = vec2 (mod (tCur, 10.) - 20., 0.6 * dir);
  vr = vec2 (0.);
  for (int k = 0; k < 30; k ++) {
    vr += dir * VortF (4. * q, c);
    c = vec2 (c.x + 1., - c.y);
    dir = - dir;
  }
  return vr;
}

void main()
{
  vec2 uv = gl_FragCoord.xy / iResolution.xy - 0.5;
  uv.x *= iResolution.x / iResolution.y;
  tCur = iTime;
  vec2 p = uv;
  for (int i = 0; i < 10; i ++) p -= FlowField (p) * 0.03;
  vec3 col = Fbm2 (5. * p + vec2 (-0.1 * tCur, 0.)) *
     vec3 (0.5, 0.5, 1.);
  fragColor = vec4 (col, 1.);
}


"""
# Referencia https://www.shadertoy.com/view/Xs2SWd
fragment_shader5 = """
#version 460

vec2 iResolution = vec2(2, 2);
layout (location = 0) out vec4 fragColor;
in vec2 fragCoord;
uniform float iTime;
float gTime;

/**
 * Created by Kamil Kolaczynski (revers) - 2014

 * Modified version of shader "abstarct" ( https://www.shadertoy.com/view/4sSGDd ) by avix.
 */
#define NEAR 0.0
#define FAR 50.0
#define MAX_STEPS 64

#define PI 3.14159265359
#define EPS 0.001

// Hash by iq
float hash(vec2 p) {
	float h = 1.0 + dot(p, vec2(127.1, 311.7));
	return fract(sin(h) * 43758.5453123);
}

float rbox(vec3 p, vec3 s, float r) {
	return length(max(abs(p) - s + vec3(r), 0.0)) - r;
}

vec2 rot(vec2 k, float t) {
	float ct = cos(t);
	float st = sin(t);
	return vec2(ct * k.x - st * k.y, st * k.x + ct * k.y);
}

void oprep2(inout vec2 p, float q, float s, float k) {
	float r = 1.0 / q;
	float ofs = s;
	float angle = atan(p.x, p.y);
	float a = mod(angle, 2.0 * PI * r) - PI * r;
	p.xy = vec2(sin(a), cos(a)) * length(p.xy) - ofs;
	p.x += ofs;
}

float map(vec3 p) {
	p.y -= 1.0;
	p.xy = rot(p.xy, p.z * 0.15);
	p.z += iTime;
	p.xy = mod(p.xy, 6.0) - 0.5 * 6.0;
	p.xy = rot(p.xy, -floor(p.z / 0.75) * 0.35);
	p.z = mod(p.z, 0.75) - 0.5 * 0.75;
	oprep2(p.xy, 6.0, 0.45, iTime);

	return rbox(p, vec3(0.1, 0.025, 0.25), 0.05);
}

vec3 getNormal(vec3 p) {
	float h = 0.0001;

	return normalize(
			vec3(map(p + vec3(h, 0, 0)) - map(p - vec3(h, 0, 0)),
					map(p + vec3(0, h, 0)) - map(p - vec3(0, h, 0)),
					map(p + vec3(0, 0, h)) - map(p - vec3(0, 0, h))));
}

float saw(float x, float d, float s, float shift) {
	float xp = PI * (x * d + iTime * 0.5 + shift);

	float as = asin(s);
	float train = 0.5 * sign(sin(xp - as) - s) + 0.5;

	float range = (PI - 2.0 * as);
	xp = mod(xp, 2.0 * PI);
	float y = mod(-(xp - 2.0 * as), range) / range;
	y *= train;

	return y;
}

vec3 getShading(vec3 p, vec3 normal, vec3 lightPos) {
	vec3 lightDirection = normalize(lightPos - p);
	float lightIntensity = clamp(dot(normal, lightDirection), 0.0, 1.0);

	vec2 id = floor((p.xy + 3.0) / 6.0);
	float fid = hash(id);
	float ve = hash(id);

	vec3 col = vec3(0.0, 1.0, 0.0);
	col *= 4.0 * saw(p.z, 0.092, 0.77, fid * 2.5);

	vec3 amb = vec3(0.15, 0.2, 0.32);
	vec3 tex = vec3(0.8098039, 0.8607843, 1.0);

	return col * tex * lightIntensity + amb * (1.0 - lightIntensity);
}

void raymarch(vec3 ro, vec3 rd, out int i, out float t) {
	t = 0.0;

	for (int j = 0; j < MAX_STEPS; ++j) {
		vec3 p = ro + rd * t;
		float h = map(p);
		i = j;

		if (h < EPS || t > FAR) {
			break;
		}
		t += h * 0.7;
	}
}

float computeSun(vec3 ro, vec3 rd, float t, float lp) {
	vec3 lpos = vec3(0.0, 0.0, 54.0);
	ro -= lpos;
	float m = dot(rd, -ro);
	float d = length(ro - vec3(0.0, 0.0, 0.7) + m * rd);

	float a = -m;
	float b = t - m;
	float aa = atan(a / d);
	float ba = atan(b / d);
	float to = (ba - aa) / d;

	return to * 0.15 * lp;
}

vec3 computeColor(vec3 ro, vec3 rd) {
	int i;
	float t;
	raymarch(ro, rd, i, t);

	float lp = sin(iTime - 1.0) + 1.3;
	vec3 color = vec3(0.0, 1.0, 0.0);

	if (i < MAX_STEPS && t >= NEAR && t <= FAR) {
		vec3 p = ro + rd * t;
		vec3 normal = getNormal(p);

		float z = 1.0 - (NEAR + t) / (FAR - NEAR);

		color = getShading(p, normal, vec3(0.0));
		color *= lp;

		float zSqrd = z * z;
		color = mix(vec3(0.0), color, zSqrd * (3.0 - 2.0 * z)); // Fog

		color += computeSun(ro, rd, t, lp);
		return pow(color, vec3(0.8));
	}
	return color * computeSun(ro, rd, t, lp);
}

void main() {
	vec2 q = fragCoord.xy / iResolution.xy;
	vec2 coord = 2.0 * q - 1.0;
	coord.x *= iResolution.x / iResolution.y;
	coord *= 0.84;

	vec3 dir = vec3(0.0, 0.0, 1.0);
	vec3 up = vec3(0.0, 1.0, 0.0);

	vec3 right = normalize(cross(dir, up));

	vec3 ro = vec3(0.0, 0.0, 8.74);
	vec3 rd = normalize(dir * 2.0 + coord.x * right + coord.y * up);
	vec3 col = computeColor(ro, rd);

	fragColor = vec4(col, 1.0);
}

"""
# Los diferentes shaders fueron obtenidos de https://www.shadertoy.com/

compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)

compiled_fragment_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
compiled_fragment_shader2 = compileShader(fragment_shader2, GL_FRAGMENT_SHADER)
compiled_fragment_shader3 = compileShader(fragment_shader3, GL_FRAGMENT_SHADER)
compiled_fragment_shader4 = compileShader(fragment_shader4, GL_FRAGMENT_SHADER)
compiled_fragment_shader5 = compileShader(fragment_shader5, GL_FRAGMENT_SHADER)

shader = compileProgram(compiled_vertex_shader, compiled_fragment_shader)
shader2 = compileProgram(compiled_vertex_shader, compiled_fragment_shader2)
shader3 = compileProgram(compiled_vertex_shader, compiled_fragment_shader3)
shader4 = compileProgram(compiled_vertex_shader, compiled_fragment_shader4)
shader5 = compileProgram(compiled_vertex_shader, compiled_fragment_shader5)

glUseProgram(shader)
glEnable(GL_DEPTH_TEST)

obj = Obj("coffe.obj")

# Se cargan los objetos para vertices
vertex = []
for ver in obj.vertices:
    for v in ver:
        vertex.append(v)

vertex_data = numpy.array(vertex, dtype=numpy.float32)

# Se cargan los objetos para caras
faces = []
for face in obj.faces:
    for f in face:
        faces.append(int(f[0]) - 1)

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


def calculateMatrix(angle, new_vec):
    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, -0.5, 0))
    rotate = glm.rotate(i, glm.radians(angle), new_vec)
    scale = glm.scale(i, glm.vec3(1, 1, 1))

    model = translate * rotate * scale

    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    projection = glm.perspective(glm.radians(45), 1600 / 1200, 0.1, 1000.0)

    glViewport(0, 0, 1000, 800)

    amatrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader, "amatrix"), 1, GL_FALSE, glm.value_ptr(amatrix)
    )


running = True

glClearColor(0, 0, 0, 1.0)

angle = 0
new_vec = glm.vec3(0, 1, 0)
prev_time = pygame.time.get_ticks()

current_shader = shader
while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(current_shader)

    glUniform1f(glGetUniformLocation(shader, "iTime"), angle / 1000)

    glDrawElements(GL_TRIANGLES, len(index_data), GL_UNSIGNED_INT, None)

    calculateMatrix(angle, new_vec)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                current_shader = shader
            if event.key == pygame.K_2:
                current_shader = shader2
            if event.key == pygame.K_3:
                current_shader = shader3
            if event.key == pygame.K_4:
                current_shader = shader4
            if event.key == pygame.K_5:
                current_shader = shader5
            if event.key == pygame.K_w:
                new_vec = glm.vec3(1, 0, 0)
                angle += 10
            if event.key == pygame.K_s:
                new_vec = glm.vec3(1, 0, 0)
                angle -= 10
            if event.key == pygame.K_d:
                new_vec = glm.vec3(0, 1, 0)
                angle += 10
            if event.key == pygame.K_a:
                new_vec = glm.vec3(0, 1, 0)
                angle -= 10
