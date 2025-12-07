"""
DETECTOR DE HEAT STAKES - Clustering de Cilindros (Optimizado para piezas ensambladas)
Detecta heat stakes en archivos STEP usando clustering de cilindros.

------------------------------------------------------------
Developer: Pablo
Contacto:  codewithpablo@gmail.com
------------------------------------------------------------
"""

import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepTools import breptools_UVBounds
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import time

class HeatStakeDetector:
    """Detector de heat stakes para piezas ensambladas"""
    
    def __init__(self, step_file):
        self.step_file = step_file
        self.shape = None
        self.cylinders = []
        self.heat_stakes = []
        
        # CRITERIO PRINCIPAL: Número de cilindros por heat stake
        self.TARGET_CYLINDERS = 7
        self.CYLINDER_TOLERANCE = 2  # Acepta de 5 a 9 cilindros
        
        # Parámetros de clustering
        self.DEFAULT_EPS = 25.0  # Reducido: cilindros de un heat stake están MUY cerca
        self.DEFAULT_MIN_SAMPLES = 5  # Mínimo de cilindros para considerar un grupo
        
    def load_step(self):
        """Carga el archivo STEP"""
        print(f"\n📂 Cargando archivo: {self.step_file}")
        reader = STEPControl_Reader()
        status = reader.ReadFile(self.step_file)
        
        if status != 1:
            raise Exception("❌ Error al leer el archivo STEP")
        
        reader.TransferRoots()
        self.shape = reader.OneShape()
        print("✓ Archivo cargado correctamente")
        return self.shape
    
    def extract_cylinders(self):
        """Extrae todas las caras cilíndricas con sus propiedades"""
        print("\n🔍 Extrayendo cilindros del modelo...")
        
        explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        cylinder_count = 0
        
        while explorer.More():
            face = explorer.Current()
            surf = BRepAdaptor_Surface(face)
            
            if surf.GetType() == GeomAbs_Cylinder:
                # Obtener propiedades del cilindro
                cylinder_geom = surf.Cylinder()
                axis = cylinder_geom.Axis()
                location = axis.Location()
                direction = axis.Direction()
                radius = cylinder_geom.Radius()
                
                # Obtener bounds UV
                u_min, u_max, v_min, v_max = breptools_UVBounds(face)
                height = abs(v_max - v_min)
                
                # Calcular área
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face, props)
                area = props.Mass()
                
                cylinder_info = {
                    'face': face,
                    'center': (location.X(), location.Y(), location.Z()),
                    'direction': (direction.X(), direction.Y(), direction.Z()),
                    'radius': radius,
                    'height': height,
                    'area': area
                }
                
                self.cylinders.append(cylinder_info)
                cylinder_count += 1
            
            explorer.Next()
        
        print(f"✓ Encontrados {cylinder_count} cilindros")
        return self.cylinders
    
    def cluster_cylinders(self, eps=None, min_samples=None):
        """Agrupa cilindros cercanos usando DBSCAN"""
        
        if eps is None:
            eps = self.DEFAULT_EPS
        if min_samples is None:
            min_samples = self.DEFAULT_MIN_SAMPLES
            
        print(f"\n🔬 Agrupando cilindros cercanos...")
        print(f"   Parámetros: eps={eps:.1f}mm, min_samples={min_samples}")
        
        if len(self.cylinders) == 0:
            print("⚠️ No hay cilindros para agrupar")
            return [], []
        
        # Extraer coordenadas de centros
        centers = np.array([cyl['center'] for cyl in self.cylinders])
        
        # Aplicar DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(centers)
        
        # Organizar por cluster
        clusters = defaultdict(list)
        noise_count = 0
        
        for idx, label in enumerate(labels):
            if label == -1:
                noise_count += 1
            else:
                clusters[label].append(idx)
        
        n_clusters = len(clusters)
        print(f"✓ Formados {n_clusters} clusters")
        print(f"   Cilindros sin grupo (ruido): {noise_count}")
        
        return clusters, labels
    
    def analyze_cluster(self, cluster_indices):
        """Analiza un cluster para caracterizarlo"""
        
        cluster_cylinders = [self.cylinders[i] for i in cluster_indices]
        num_cylinders = len(cluster_cylinders)
        
        # Calcular centroide
        centers = np.array([cyl['center'] for cyl in cluster_cylinders])
        centroid = centers.mean(axis=0)
        
        # Calcular spread (dispersión espacial)
        distances = np.linalg.norm(centers - centroid, axis=1)
        max_spread = distances.max()
        avg_spread = distances.mean()
        
        # Calcular bounding box
        min_coords = centers.min(axis=0)
        max_coords = centers.max(axis=0)
        dimensions = max_coords - min_coords
        bbox_volume = np.prod(dimensions) if np.all(dimensions > 0) else 0
        
        # Estadísticas de radios
        radii = [cyl['radius'] for cyl in cluster_cylinders]
        avg_radius = np.mean(radii)
        std_radius = np.std(radii)
        
        # Estadísticas de alturas
        heights = [cyl['height'] for cyl in cluster_cylinders]
        avg_height = np.mean(heights)
        
        analysis = {
            'num_cylinders': num_cylinders,
            'centroid': tuple(centroid),
            'max_spread': max_spread,
            'avg_spread': avg_spread,
            'dimensions': tuple(dimensions),
            'bbox_volume': bbox_volume,
            'avg_radius': avg_radius,
            'std_radius': std_radius,
            'avg_height': avg_height,
            'cylinder_indices': cluster_indices,
            'cylinders': cluster_cylinders
        }
        
        return analysis
    
    def is_heat_stake(self, cluster_analysis):
        """
        Valida si un cluster es un heat stake
        CRITERIO PRINCIPAL: Número de cilindros cercanos
        """
        
        num_cyl = cluster_analysis['num_cylinders']
        spread = cluster_analysis['max_spread']
        
        # VALIDACIÓN 1: Número de cilindros (CRÍTICO)
        cyl_min = self.TARGET_CYLINDERS - self.CYLINDER_TOLERANCE
        cyl_max = self.TARGET_CYLINDERS + self.CYLINDER_TOLERANCE
        cyl_valid = cyl_min <= num_cyl <= cyl_max
        
        # VALIDACIÓN 2: Los cilindros deben estar compactos (no dispersos)
        # Heat stakes típicos tienen cilindros en un área de ~50mm
        spread_valid = spread < 100.0  # mm
        
        # VALIDACIÓN 3: Radio promedio razonable (opcional)
        avg_radius = cluster_analysis['avg_radius']
        radius_valid = 0.5 < avg_radius < 10.0  # mm
        
        # Decisión final
        is_valid = cyl_valid and spread_valid and radius_valid
        
        validation = {
            'is_heat_stake': is_valid,
            'cylinder_check': cyl_valid,
            'spread_check': spread_valid,
            'radius_check': radius_valid,
            'confidence': 'HIGH' if (cyl_valid and spread_valid) else 'LOW',
            'details': {
                'cylinders': f"{num_cyl} (target: {self.TARGET_CYLINDERS}±{self.CYLINDER_TOLERANCE})",
                'spread': f"{spread:.2f}mm (max: 100mm)",
                'avg_radius': f"{avg_radius:.2f}mm"
            }
        }
        
        return validation
    
    def detect_heat_stakes(self, eps=None, min_samples=None, show_all_clusters=False):
        """Proceso completo de detección"""
        
        print("\n" + "="*80)
        print("🎯 DETECTOR DE HEAT STAKES - Optimizado para Piezas Ensambladas")
        print("="*80)
        
        start_time = time.time()
        
        # 1. Cargar archivo
        self.load_step()
        
        # 2. Extraer cilindros
        self.extract_cylinders()
        
        if len(self.cylinders) == 0:
            print("\n⚠️ No se encontraron cilindros")
            return []
        
        # 3. Agrupar cilindros
        clusters, labels = self.cluster_cylinders(eps=eps, min_samples=min_samples)
        
        if len(clusters) == 0:
            print("\n⚠️ No se formaron clusters")
            print("💡 Intenta ajustar los parámetros:")
            print("   - Aumentar eps (ej: 30-40mm)")
            print("   - Reducir min_samples (ej: 3-4)")
            return []
        
        # 4. Analizar cada cluster
        print(f"\n📊 Analizando {len(clusters)} clusters...")
        print("="*80)
        
        heat_stake_candidates = []
        rejected_clusters = []
        
        for cluster_id, indices in sorted(clusters.items()):
            
            # Analizar cluster
            analysis = self.analyze_cluster(indices)
            
            # Validar si es heat stake
            validation = self.is_heat_stake(analysis)
            
            if validation['is_heat_stake']:
                heat_stake_candidates.append({
                    'cluster_id': cluster_id,
                    'analysis': analysis,
                    'validation': validation
                })
                
                # Mostrar solo heat stakes detectados
                print(f"\n✅ Heat Stake #{len(heat_stake_candidates)} (Cluster #{cluster_id + 1}):")
                print(f"   Cilindros: {analysis['num_cylinders']}")
                print(f"   Centroide: ({analysis['centroid'][0]:.2f}, {analysis['centroid'][1]:.2f}, {analysis['centroid'][2]:.2f})")
                print(f"   Dispersión: {analysis['max_spread']:.2f}mm")
                print(f"   Radio promedio: {analysis['avg_radius']:.2f}mm")
                print(f"   Confianza: {validation['confidence']}")
            else:
                rejected_clusters.append({
                    'cluster_id': cluster_id,
                    'analysis': analysis,
                    'validation': validation
                })
        
        # Mostrar clusters rechazados (opcional)
        if show_all_clusters and rejected_clusters:
            print(f"\n" + "-"*80)
            print(f"❌ Clusters rechazados ({len(rejected_clusters)}):")
            for item in rejected_clusters[:5]:  # Mostrar solo primeros 5
                cid = item['cluster_id']
                ana = item['analysis']
                val = item['validation']
                print(f"\n   Cluster #{cid + 1}:")
                print(f"   Cilindros: {'✓' if val['cylinder_check'] else '✗'} {val['details']['cylinders']}")
                print(f"   Dispersión: {'✓' if val['spread_check'] else '✗'} {val['details']['spread']}")
        
        # 5. Resumen final
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("📋 RESUMEN")
        print("="*80)
        print(f"Total cilindros: {len(self.cylinders)}")
        print(f"Clusters formados: {len(clusters)}")
        print(f"Heat stakes detectados: {len(heat_stake_candidates)}")
        print(f"Tiempo: {elapsed:.2f}s")
        
        if len(heat_stake_candidates) > 0:
            print(f"\n🎯 COORDENADAS DE HEAT STAKES:")
            print("-"*80)
            for i, hs in enumerate(heat_stake_candidates):
                c = hs['analysis']['centroid']
                print(f"  HS#{i+1:2d}: X={c[0]:8.2f}, Y={c[1]:8.2f}, Z={c[2]:8.2f}")
        
        # Análisis de discrepancia
        expected = 29  # Conteo manual del usuario
        detected = len(heat_stake_candidates)
        if detected != expected:
            print(f"\n⚠️  NOTA: Detectados {detected} vs {expected} esperados")
            if detected < expected:
                print("💡 Para detectar más heat stakes, intenta:")
                print("   python detector.py archivo.step 30.0 4")
                print("                                    ↑    ↑")
                print("                                    |    min cilindros")
                print("                                    distancia (mm)")
            elif detected > expected:
                print("💡 Para ser más estricto, intenta:")
                print("   python detector.py archivo.step 20.0 6")
        
        print("="*80)
        
        self.heat_stakes = heat_stake_candidates
        return heat_stake_candidates


def main():
    """Función principal"""
    
    if len(sys.argv) < 2:
        print("="*80)
        print("DETECTOR DE HEAT STAKES - Uso")
        print("="*80)
        print("\n📝 Sintaxis:")
        print("   python detector.py <archivo.step> [eps] [min_samples] [--all]")
        print("\n📌 Ejemplos:")
        print("   python detector.py modelo.step")
        print("   python detector.py modelo.step 30.0 5")
        print("   python detector.py modelo.step 25.0 4 --all")
        print("\n⚙️  Parámetros:")
        print("   eps           Distancia máxima entre cilindros (default: 25.0mm)")
        print("   min_samples   Mínimo cilindros por grupo (default: 5)")
        print("   --all         Mostrar también clusters rechazados")
        print("\n💡 Tips:")
        print("   - Si detecta menos heat stakes, REDUCE eps (20-25mm)")
        print("   - Si detecta más de lo esperado, AUMENTA min_samples (6-7)")
        print("="*80)
        sys.exit(1)
    
    step_file = sys.argv[1]
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else None
    min_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
    show_all = '--all' in sys.argv
    
    # Crear y ejecutar detector
    detector = HeatStakeDetector(step_file)
    heat_stakes = detector.detect_heat_stakes(
        eps=eps, 
        min_samples=min_samples,
        show_all_clusters=show_all
    )
    
    # Exportar coordenadas
    if heat_stakes:
        output_file = 'heat_stakes_coordinates.txt'
        print(f"\n💾 Exportando a '{output_file}'...")
        
        with open(output_file, 'w') as f:
            f.write("COORDENADAS DE HEAT STAKES DETECTADOS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total detectados: {len(heat_stakes)}\n")
            f.write(f"Archivo: {step_file}\n\n")
            f.write("-"*80 + "\n")
            
            for i, hs in enumerate(heat_stakes):
                c = hs['analysis']['centroid']
                n = hs['analysis']['num_cylinders']
                s = hs['analysis']['max_spread']
                
                f.write(f"\nHeat Stake #{i+1}:\n")
                f.write(f"  Centroide: X={c[0]:.4f}, Y={c[1]:.4f}, Z={c[2]:.4f}\n")
                f.write(f"  Cilindros: {n}\n")
                f.write(f"  Dispersión: {s:.2f}mm\n")
        
        print(f"✓ Archivo creado")


if __name__ == "__main__":
    main()