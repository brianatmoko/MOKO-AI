"""
MOKO Spatial & Graphics Math Reasoning Engine (SGM Engine)
==========================================================
Engine matematika kognitif tingkat tinggi untuk 3D, Grafika Komputer, Kinematika Kamera,
dan Mekanika Game Real-Time.

Komponen Utama:
1. Vector3D & Matrix4x4: Aljabar Linier 3D & Matriks Transformasi (MVP / LookAt).
2. Quaternion: Rotasi 3D bebas gimbal-lock & interpolasi slerp() untuk kamera.
3. Raycast3D: Interseksi Ray-Sphere, Ray-Plane, dan Ray-Triangle (Möller-Trumbore).
4. CurveSpline: Kurva Bezier Kubik & Catmull-Rom Spline untuk trajektori & kamera move.
5. CodeSnippetsGenerator: Helper generator JS/Python murni untuk disuntikkan ke LLM.
"""

import math
from typing import Tuple, List, Dict, Any, Optional


class Vector3D:
    """Representasi Vektor 3-Dimensi"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vector3D':
        return self.__mul__(scalar)

    def dot(self, other: 'Vector3D') -> float:
        """Dot Product (Perkalian Titik) -> Mengukur sudut / proyeksi"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross Product (Perkalian Silang) -> Vektor tegak lurus (Normal)"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        """Panjang / Magnitude Vektor"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'Vector3D':
        """Vektor Satuan (Unit Vector) dengan panjang 1"""
        l = self.length()
        if l > 1e-9:
            return Vector3D(self.x / l, self.y / l, self.z / l)
        return Vector3D(0, 0, 0)

    @staticmethod
    def lerp(v1: 'Vector3D', v2: 'Vector3D', t: float) -> 'Vector3D':
        """Linear Interpolation antara dua vektor"""
        t = max(0.0, min(1.0, t))
        return v1 * (1.0 - t) + v2 * t

    def to_list(self) -> List[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.z, 4)]

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Matrix4x4:
    """Matriks Transformasi 4x4 untuk Grafika 3D (Model, View, Projection)"""

    def __init__(self, data: Optional[List[List[float]]] = None):
        if data:
            self.m = data
        else:
            # Identity Matrix default
            self.m = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]

    def multiply(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Perkalian Matriks 4x4"""
        res = [[0.0] * 4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                res[r][c] = sum(self.m[r][k] * other.m[k][c] for k in range(4))
        return Matrix4x4(res)

    def transform_vector(self, v: Vector3D) -> Vector3D:
        """Proyeksi/Transformasi Vektor 3D oleh Matriks 4x4 (w=1)"""
        x = self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z + self.m[0][3]
        y = self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z + self.m[1][3]
        z = self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z + self.m[2][3]
        w = self.m[3][0]*v.x + self.m[3][1]*v.y + self.m[3][2]*v.z + self.m[3][3]
        if abs(w) > 1e-9 and abs(w - 1.0) > 1e-9:
            return Vector3D(x / w, y / w, z / w)
        return Vector3D(x, y, z)

    @staticmethod
    def translation(tx: float, ty: float, tz: float) -> 'Matrix4x4':
        return Matrix4x4([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def scale(sx: float, sy: float, sz: float) -> 'Matrix4x4':
        return Matrix4x4([
            [sx,  0.0, 0.0, 0.0],
            [0.0, sy,  0.0, 0.0],
            [0.0, 0.0, sz,  0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def look_at(eye: Vector3D, target: Vector3D, up: Vector3D) -> 'Matrix4x4':
        """Matriks Kamera View (LookAt)"""
        z_axis = (eye - target).normalize()  # Forward
        x_axis = up.cross(z_axis).normalize() # Right
        y_axis = z_axis.cross(x_axis)        # Up real
        
        return Matrix4x4([
            [x_axis.x, x_axis.y, x_axis.z, -x_axis.dot(eye)],
            [y_axis.x, y_axis.y, y_axis.z, -y_axis.dot(eye)],
            [z_axis.x, z_axis.y, z_axis.z, -z_axis.dot(eye)],
            [0.0,      0.0,      0.0,      1.0]
        ])


class Quaternion:
    """Quaternion (w, x, y, z) untuk Rotasi 3D Tanpa Gimbal Lock"""

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def from_euler(pitch_rad: float, yaw_rad: float, roll_rad: float) -> 'Quaternion':
        """Konversi sudut Euler (pitch, yaw, roll) ke Quaternion"""
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return Quaternion(w, x, y, z)

    def multiply(self, q: 'Quaternion') -> 'Quaternion':
        """Perkalian dua Quaternion (Menggabungkan rotasi)"""
        w = self.w*q.w - self.x*q.x - self.y*q.y - self.z*q.z
        x = self.w*q.x + self.x*q.w + self.y*q.z - self.z*q.y
        y = self.w*q.y - self.x*q.z + self.y*q.w + self.z*q.x
        z = self.w*q.z + self.x*q.y - self.y*q.x + self.z*q.w
        return Quaternion(w, x, y, z)

    def normalize(self) -> 'Quaternion':
        n = math.sqrt(self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)
        if n > 1e-9:
            return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
        return Quaternion(1, 0, 0, 0)

    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical Linear Interpolation (SLERP) — Pergerakan Kamera 3D Mulus"""
        q1 = q1.normalize()
        q2 = q2.normalize()

        dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
        
        # Jika dot product negatif, balik q2 agar mengambil jalur terpendek
        if dot < 0.0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot

        if dot > 0.9995:
            # Jika sudut sangat kecil, gunakan LERP biasa untuk stabilitas numerik
            return Quaternion(
                q1.w + t*(q2.w - q1.w),
                q1.x + t*(q2.x - q1.x),
                q1.y + t*(q2.y - q1.y),
                q1.z + t*(q2.z - q1.z)
            ).normalize()

        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        return Quaternion(
            s1*q1.w + s2*q2.w,
            s1*q1.x + s2*q2.x,
            s1*q1.y + s2*q2.y,
            s1*q1.z + s2*q2.z
        )

    def __repr__(self) -> str:
        return f"Quat(w={self.w:.3f}, x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class Raycast3D:
    """Mekanika Interseksi Raycasting 3D (Collision, 3D Selection)"""

    @staticmethod
    def ray_sphere_intersection(ray_origin: Vector3D, ray_dir: Vector3D, sphere_center: Vector3D, sphere_radius: float) -> Optional[float]:
        """
        Hitung titik t interseksi Ray-Sphere.
        Kembalikan t (jarak dari origin) atau None jika tidak bertabrakan.
        """
        d = ray_dir.normalize()
        oc = ray_origin - sphere_center
        b = 2.0 * oc.dot(d)
        c = oc.dot(oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4.0 * c
        
        if discriminant < 0.0:
            return None
        t = (-b - math.sqrt(discriminant)) / 2.0
        if t < 0.0:
            t = (-b + math.sqrt(discriminant)) / 2.0
        return t if t >= 0.0 else None

    @staticmethod
    def ray_triangle_moller_trumbore(ray_origin: Vector3D, ray_dir: Vector3D, v0: Vector3D, v1: Vector3D, v2: Vector3D) -> Optional[Tuple[float, float, float]]:
        """
        Algoritma Möller–Trumbore: Interseksi Ray-Segitiga 3D Realtime.
        Returns: (t, u, v) koordinat barisentrik jika bertabrakan.
        """
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = ray_dir.cross(edge2)
        a = edge1.dot(h)

        if -1e-7 < a < 1e-7:
            return None  # Ray sejajar dengan segitiga

        f = 1.0 / a
        s = ray_origin - v0
        u = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            return None

        q = s.cross(edge1)
        v = f * ray_dir.dot(q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * edge2.dot(q)
        if t > 1e-7:
            return (t, u, v)
        return None


class CurveSpline:
    """Interpolasi Kurva Spline untuk Gerakan Kamera & Trajektori Parabola Game"""

    @staticmethod
    def cubic_bezier(p0: Vector3D, p1: Vector3D, p2: Vector3D, p3: Vector3D, t: float) -> Vector3D:
        """Hitung posisi pada Kurva Bezier Kubik pada t ∈ [0, 1]"""
        t = max(0.0, min(1.0, t))
        u = 1.0 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t

        p = uuu * p0
        p = p + (3.0 * uu * t) * p1
        p = p + (3.0 * u * tt) * p2
        p = p + ttt * p3
        return p

    @staticmethod
    def catmull_rom(p0: Vector3D, p1: Vector3D, p2: Vector3D, p3: Vector3D, t: float) -> Vector3D:
        """Interpolasi Catmull-Rom Spline mulus melintasi titik-titik kontrol p1 ke p2"""
        t = max(0.0, min(1.0, t))
        t2 = t * t
        t3 = t2 * t

        f0 = -0.5 * t3 + t2 - 0.5 * t
        f1 = 1.5 * t3 - 2.5 * t2 + 1.0
        f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        f3 = 0.5 * t3 - 0.5 * t2

        return p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3


class SGMEngine:
    """Spatial & Graphics Math Reasoning Engine — Main Interface"""

    @staticmethod
    def generate_js_math_helpers() -> str:
        """
        Menghasilkan pustaka helper matematika 3D murni JavaScript 
        siap disuntikkan ke proyek WebGL / Canvas / Web Game.
        """
        return """
// === MOKO SGM ENGINE: JAVASCRIPT SPATIAL MATH HELPERS ===
class Vec3 {
    constructor(x=0, y=0, z=0) { this.x = x; this.y = y; this.z = z; }
    add(v) { return new Vec3(this.x+v.x, this.y+v.y, this.z+v.z); }
    sub(v) { return new Vec3(this.x-v.x, this.y-v.y, this.z-v.z); }
    scale(s) { return new Vec3(this.x*s, this.y*s, this.z*s); }
    dot(v) { return this.x*v.x + this.y*v.y + this.z*v.z; }
    cross(v) { return new Vec3(this.y*v.z - this.z*v.y, this.z*v.x - this.x*v.z, this.x*v.y - this.y*v.x); }
    length() { return Math.sqrt(this.dot(this)); }
    normalize() { let l = this.length(); return l > 1e-7 ? this.scale(1/l) : new Vec3(); }
    static lerp(v1, v2, t) { return v1.scale(1-t).add(v2.scale(t)); }
}

class Quat {
    constructor(w=1, x=0, y=0, z=0) { this.w = w; this.x = x; this.y = y; this.z = z; }
    static slerp(q1, q2, t) {
        let dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z;
        if (dot < 0) { q2 = new Quat(-q2.w, -q2.x, -q2.y, -q2.z); dot = -dot; }
        if (dot > 0.9995) return new Quat(q1.w+t*(q2.w-q1.w), q1.x+t*(q2.x-q1.x), q1.y+t*(q2.y-q1.y), q1.z+t*(q2.z-q1.z));
        let theta_0 = Math.acos(Math.max(-1, Math.min(1, dot))), theta = theta_0*t;
        let s1 = Math.cos(theta) - dot*Math.sin(theta)/Math.sin(theta_0), s2 = Math.sin(theta)/Math.sin(theta_0);
        return new Quat(s1*q1.w + s2*q2.w, s1*q1.x + s2*q2.x, s1*q1.y + s2*q2.y, s1*q1.z + s2*q2.z);
    }
}
"""

    @staticmethod
    def evaluate_spatial_query(query: str) -> Dict[str, Any]:
        """Evaluasi kueri spasial matematika secara deterministik"""
        q = query.lower()
        if "quaternion" in q or "slerp" in q:
            q1 = Quaternion.from_euler(0, 0, 0)
            q2 = Quaternion.from_euler(0, math.pi / 2, 0)
            mid = Quaternion.slerp(q1, q2, 0.5)
            return {
                "type": "quaternion_slerp",
                "result": str(mid),
                "explanation": "Interpolasi Slerp 50% antara Yaw 0 dan Yaw 90 derajat."
            }
        elif "raycast" in q or "interseksi" in q or "collision" in q:
            hit = Raycast3D.ray_sphere_intersection(Vector3D(0,0,-5), Vector3D(0,0,1), Vector3D(0,0,0), 2.0)
            return {
                "type": "raycast_sphere",
                "hit_distance": hit,
                "explanation": f"Sinar dari (0,0,-5) menghantam bola r=2.0 pada jarak t={hit:.3f}" if hit else "Tidak bertabrakan."
            }
        return {
            "type": "general_3d_math",
            "result": "SGM Engine Ready.",
            "explanation": "Penalaran aljabar linier spasial aktif."
        }


# Singleton
sgm_engine = SGMEngine()
