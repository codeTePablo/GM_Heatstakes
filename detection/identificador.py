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
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shape
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Display.OCCViewer import Viewer3d
from OCC.Display.SimpleGui import init_display
from OCC.Core.AIS import AIS_TextLabel
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
        self.rejected_clusters = []  # NUEVO: Guardar clusters rechazados
        
        # CRITERIO FLEXIBLE: Número de cilindros por heat stake
        # Basado en observaciones: heat stakes pueden tener 7-20 cilindros
        self.MIN_CYLINDERS = 5   # Mínimo absoluto
        self.MAX_CYLINDERS = 25  # Máximo absoluto
        self.TARGET_CYLINDERS = 7  # Referencia ideal
        
        # Parámetros de clustering
        self.DEFAULT_EPS = 25.0  # Distancia para agrupar cilindros
        self.DEFAULT_MIN_SAMPLES = 5  # Mínimo de cilindros para formar grupo
        
        # Modo de análisis
        self.STRICT_MODE = False  # Si True, usa criterio estricto (7±2)
        
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
        MODO FLEXIBLE: Acepta rangos más amplios de cilindros
        """
        
        num_cyl = cluster_analysis['num_cylinders']
        spread = cluster_analysis['max_spread']
        avg_radius = cluster_analysis['avg_radius']
        
        # VALIDACIÓN 1: Número de cilindros
        if self.STRICT_MODE:
            # Modo estricto: solo 7±2 cilindros
            cyl_valid = (self.TARGET_CYLINDERS - 2) <= num_cyl <= (self.TARGET_CYLINDERS + 2)
            confidence = 'HIGH' if cyl_valid else 'LOW'
        else:
            # Modo flexible: acepta rangos más amplios
            if self.MIN_CYLINDERS <= num_cyl <= self.MAX_CYLINDERS:
                cyl_valid = True
                # Calcular nivel de confianza basado en cercanía al target
                diff = abs(num_cyl - self.TARGET_CYLINDERS)
                if diff <= 2:
                    confidence = 'HIGH'
                elif diff <= 5:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
            else:
                cyl_valid = False
                confidence = 'REJECTED'
        
        # VALIDACIÓN 2: Dispersión espacial (los cilindros deben estar compactos)
        # Heat stakes típicos: cilindros en área compacta
        spread_valid = spread < 100.0  # mm
        
        # VALIDACIÓN 3: Radio promedio razonable
        radius_valid = 0.5 < avg_radius < 10.0  # mm
        
        # VALIDACIÓN 4: Densidad de cilindros (cilindros por mm³)
        # Si están MUY compactos, es más probable que sea un heat stake
        bbox_volume = cluster_analysis['bbox_volume']
        if bbox_volume > 0:
            density = num_cyl / bbox_volume * 1000  # normalizado
            density_valid = density > 0.1  # Al menos cierta densidad
        else:
            density_valid = True  # No penalizar si bbox es muy pequeño
        
        # Decisión final (más permisiva)
        is_valid = cyl_valid and spread_valid and radius_valid and density_valid
        
        validation = {
            'is_heat_stake': is_valid,
            'cylinder_check': cyl_valid,
            'spread_check': spread_valid,
            'radius_check': radius_valid,
            'density_check': density_valid,
            'confidence': confidence,
            'details': {
                'cylinders': f"{num_cyl} (range: {self.MIN_CYLINDERS}-{self.MAX_CYLINDERS})",
                'spread': f"{spread:.2f}mm (max: 100mm)",
                'avg_radius': f"{avg_radius:.2f}mm",
                'density': f"{num_cyl / max(bbox_volume, 1) * 1000:.3f} cyl/mm³"
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
                print(f"   Densidad: {analysis['num_cylinders'] / max(analysis['bbox_volume'], 1) * 1000:.3f} cyl/mm³")
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
            for item in rejected_clusters[:10]:  # Mostrar hasta 10
                cid = item['cluster_id']
                ana = item['analysis']
                val = item['validation']
                print(f"\n   Cluster #{cid + 1}:")
                print(f"   Cilindros: {'✓' if val['cylinder_check'] else '✗'} {val['details']['cylinders']}")
                print(f"   Dispersión: {'✓' if val['spread_check'] else '✗'} {val['details']['spread']}")
                print(f"   Radio: {'✓' if val['radius_check'] else '✗'} {val['details']['avg_radius']}")
                print(f"   Densidad: {'✓' if val['density_check'] else '✗'} {val['details']['density']}")
        
        # Análisis de confianza
        if heat_stake_candidates:
            high_conf = sum(1 for hs in heat_stake_candidates if hs['validation']['confidence'] == 'HIGH')
            med_conf = sum(1 for hs in heat_stake_candidates if hs['validation']['confidence'] == 'MEDIUM')
            low_conf = sum(1 for hs in heat_stake_candidates if hs['validation']['confidence'] == 'LOW')
            
            if high_conf + med_conf + low_conf > 0:
                print(f"\n📊 Distribución de confianza:")
                print(f"   Alta (HIGH):   {high_conf}")
                print(f"   Media (MEDIUM): {med_conf}")
                print(f"   Baja (LOW):    {low_conf}")
        
        # 5. Resumen final
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("📋 RESUMEN")
        print("="*80)
        print(f"Total cilindros: {len(self.cylinders)}")
        print(f"Clusters formados: {len(clusters)}")
        print(f"Heat stakes detectados: {len(heat_stake_candidates)}")
        print(f"Modo de detección: {'ESTRICTO (7±2 cyl)' if self.STRICT_MODE else 'FLEXIBLE (5-25 cyl)'}")
        print(f"Tiempo: {elapsed:.2f}s")
        
        if len(heat_stake_candidates) > 0:
            print(f"\n🎯 COORDENADAS DE HEAT STAKES:")
            print("-"*80)
            print(f"{'ID':<5} {'Confianza':<10} {'X':>10} {'Y':>10} {'Z':>10} {'Cyl':>5}")
            print("-"*80)
            for i, hs in enumerate(heat_stake_candidates):
                c = hs['analysis']['centroid']
                conf = hs['validation']['confidence']
                n_cyl = hs['analysis']['num_cylinders']
                conf_icon = '🟢' if conf == 'HIGH' else '🟡' if conf == 'MEDIUM' else '🔴'
                print(f"{i+1:<5} {conf_icon} {conf:<8} {c[0]:10.2f} {c[1]:10.2f} {c[2]:10.2f} {n_cyl:5}")
        
        # Análisis de discrepancia mejorado
        expected = 29  # Conteo manual del usuario
        detected = len(heat_stake_candidates)
        if detected != expected:
            print(f"\n⚠️  NOTA: Detectados {detected} vs {expected} esperados")
            if detected < expected:
                print("💡 Para detectar MÁS heat stakes:")
                print(f"   • Aumentar eps: python detector.py archivo.step 35.0 5")
                print(f"   • Reducir min_samples: python detector.py archivo.step 25.0 3")
                print(f"   • Ver rechazados: python detector.py archivo.step --all")
            elif detected > expected:
                print("💡 Para ser más SELECTIVO:")
                print(f"   • Reducir eps: python detector.py archivo.step 20.0 5")
                print(f"   • Aumentar min_samples: python detector.py archivo.step 25.0 7")
                print(f"   • Modo estricto: Editar STRICT_MODE = True en el código")
        else:
            print(f"\n✅ PERFECTO: Detectados exactamente {expected} heat stakes!")
        
        print("="*80)
        
        self.heat_stakes = heat_stake_candidates
        self.rejected_clusters = rejected_clusters  # GUARDAR rechazados para visualización
        return heat_stake_candidates
    
    def visualize_heat_stakes(self, marker_size=5.0, show_model=True, show_rejected=False):
        """
        Visualiza el modelo 3D con los heat stakes marcados
        
        Args:
            marker_size: Tamaño de las esferas marcadoras (mm)
            show_model: Si True, muestra el modelo completo
            show_rejected: Si True, muestra clusters rechazados en negro
        """
        
        if not self.heat_stakes and not (show_rejected and self.rejected_clusters):
            print("\n⚠️ No hay heat stakes ni clusters rechazados para visualizar")
            print("   Ejecuta primero: detector.detect_heat_stakes()")
            return
        
        print("\n" + "="*80)
        print("🎨 VISUALIZACIÓN 3D - Heat Stakes + Clusters Rechazados")
        print("="*80)
        print(f"Heat stakes válidos: {len(self.heat_stakes)}")
        if show_rejected:
            print(f"Clusters rechazados: {len(self.rejected_clusters)}")
        print(f"Preparando visualización...")
        
        # Inicializar display
        display, start_display, add_menu, add_function_to_menu = init_display()
        
        # Mostrar el modelo completo (semi-transparente)
        if show_model and self.shape:
            print("✓ Renderizando modelo principal...")
            ais_shape = AIS_Shape(self.shape)
            display.Context.Display(ais_shape, True)
            display.Context.SetTransparency(ais_shape, 0.7, True)
            display.Context.SetColor(ais_shape, Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB), True)
        
        # Colores para heat stakes válidos
        valid_colors = [
            (1.0, 0.0, 0.0),  # Rojo
            (0.0, 1.0, 0.0),  # Verde
            (0.0, 0.0, 1.0),  # Azul
            (1.0, 1.0, 0.0),  # Amarillo
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
        ]
        
        # Crear esferas para heat stakes VÁLIDOS
        print("✓ Creando marcadores de heat stakes válidos...")
        for i, hs in enumerate(self.heat_stakes):
            centroid = hs['analysis']['centroid']
            confidence = hs['validation']['confidence']
            
            # Crear esfera
            pnt = gp_Pnt(centroid[0], centroid[1], centroid[2])
            
            # Tamaño según confianza
            if confidence == 'HIGH':
                radius = marker_size * 1.2  # Más grandes
            elif confidence == 'MEDIUM':
                radius = marker_size * 1.0
            else:  # LOW
                radius = marker_size * 0.8  # Más pequeños
            
            sphere = BRepPrimAPI_MakeSphere(pnt, radius).Shape()
            
            # Color según posición
            color = valid_colors[i % len(valid_colors)]
            
            # Mostrar esfera
            ais_sphere = AIS_Shape(sphere)
            display.Context.Display(ais_sphere, True)
            display.Context.SetColor(
                ais_sphere,
                Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB),
                True
            )
            
            # AGREGAR ETIQUETA DE TEXTO 3D
            label = f"HS-{i+1}"
            # Posición del texto: ligeramente arriba de la esfera
            text_pos = gp_Pnt(centroid[0], centroid[1], centroid[2] + radius * 1.5)
            text_approved = AIS_TextLabel()
            text_approved.SetText(label)
            text_approved.SetPosition(text_pos)
            text_approved.SetColor(Quantity_Color(0, 0, 0, Quantity_TOC_RGB))
            text_approved.SetHeight(radius * 1.4)   # <<< AQUI SI CAMBIA EL TAMAÑO

            display.Context.Display(text_approved, True)
        
        # Crear esferas para clusters RECHAZADOS (negro)
        if show_rejected and self.rejected_clusters:
            print("✓ Creando marcadores de clusters rechazados (negro)...")
            rejected_count = 0
            
            for j, item in enumerate(self.rejected_clusters):
                centroid = item['analysis']['centroid']
                
                # Crear esfera más pequeña en negro
                pnt = gp_Pnt(centroid[0], centroid[1], centroid[2])
                radius = marker_size * 0.6  # Más pequeño
                sphere = BRepPrimAPI_MakeSphere(pnt, radius).Shape()
                
                # Color negro para rechazados
                ais_sphere = AIS_Shape(sphere)
                display.Context.Display(ais_sphere, True)
                display.Context.SetColor(
                    ais_sphere,
                    Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB),  # Gris oscuro/negro
                    True
                )
                
                # AGREGAR ETIQUETA PARA RECHAZADOS
                label = f"R-{j+1}"
                text_pos = gp_Pnt(centroid[0], centroid[1], centroid[2] + radius * 3)
                text_rejected = AIS_TextLabel()
                text_rejected.SetText(label)
                text_rejected.SetPosition(text_pos)
                text_rejected.SetColor(Quantity_Color(0.3, 0.3, 0.3, Quantity_TOC_RGB))
                text_rejected.SetHeight(radius * 1.4)   # <<< AQUÍ TAMBIÉN SÍ CRECE

                display.Context.Display(text_rejected, True)
                
                rejected_count += 1
            
            print(f"   {rejected_count} clusters rechazados marcados")
        
        print("✓ Visualización preparada")
        print("\n" + "="*80)
        print("🖱️  CONTROLES DE VISUALIZACIÓN:")
        print("="*80)
        print("  • Click izquierdo + arrastrar  → Rotar")
        print("  • Click derecho + arrastrar    → Pan (mover)")
        print("  • Rueda del mouse              → Zoom")
        print("  • Tecla 'F'                    → Fit (ajustar vista)")
        print("  • Cerrar ventana               → Salir")
        print("="*80)
        print("\n🎯 LEYENDA DE COLORES:")
        print("="*80)
        print("  🔴🟢🔵🟡🟣 Heat stakes VÁLIDOS (colores variados)")
        print("     └─ Etiqueta: HS-1, HS-2, HS-3, ...")
        print("     └─ Tamaño grande:   Confianza ALTA")
        print("     └─ Tamaño medio:    Confianza MEDIA")
        print("     └─ Tamaño pequeño:  Confianza BAJA")
        if show_rejected:
            print("  ⚫ Clusters RECHAZADOS (negro, tamaño pequeño)")
            print("     └─ Etiqueta: R-1, R-2, R-3, ...")
        print("="*80)
        print(f"\n📊 Total marcadores: {len(self.heat_stakes)} válidos", end="")
        if show_rejected:
            print(f" + {len(self.rejected_clusters)} rechazados")
        else:
            print()
        print("\n▶️  Iniciando visualización...\n")
        
        # Ajustar vista
        display.FitAll()
        
        # Iniciar loop
        start_display()
    
    def export_visualization_script(self, output_file='visualize_results.py', include_rejected=False):
        """
        Exporta un script Python standalone para visualizar los resultados
        
        Args:
            output_file: Nombre del archivo de salida
            include_rejected: Si True, incluye clusters rechazados en el script
        """
        
        if not self.heat_stakes and not (include_rejected and self.rejected_clusters):
            print("\n⚠️ No hay heat stakes para exportar script de visualización")
            return
        
        print(f"\n💾 Exportando script de visualización a '{output_file}'...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('Script de visualización de Heat Stakes detectados\n')
            f.write(f'Archivo original: {self.step_file}\n')
            f.write(f'Heat stakes detectados: {len(self.heat_stakes)}\n')
            if include_rejected:
                f.write(f'Clusters rechazados: {len(self.rejected_clusters)}\n')
            f.write('"""\n\n')
            
            f.write('from OCC.Core.STEPControl import STEPControl_Reader\n')
            f.write('from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere\n')
            f.write('from OCC.Core.gp import gp_Pnt\n')
            f.write('from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB\n')
            f.write('from OCC.Core.AIS import AIS_Shape\n')
            f.write('from OCC.Display.SimpleGui import init_display\n\n')
            
            # Heat stakes válidos
            f.write('# Coordenadas de heat stakes VÁLIDOS\n')
            f.write('# Formato: (x, y, z, confianza)\n')
            f.write('heat_stakes = [\n')
            for hs in self.heat_stakes:
                c = hs['analysis']['centroid']
                conf = hs['validation']['confidence']
                f.write(f'    ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, "{conf}"),\n')
            f.write(']\n\n')
            
            # Clusters rechazados
            if include_rejected and self.rejected_clusters:
                f.write('# Coordenadas de clusters RECHAZADOS\n')
                f.write('rejected_clusters = [\n')
                for item in self.rejected_clusters:
                    c = item['analysis']['centroid']
                    f.write(f'    ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}),\n')
                f.write(']\n\n')
            else:
                f.write('rejected_clusters = []\n\n')
            
            f.write('# Cargar modelo\n')
            f.write(f'step_file = "{self.step_file}"\n')
            f.write('reader = STEPControl_Reader()\n')
            f.write('reader.ReadFile(step_file)\n')
            f.write('reader.TransferRoots()\n')
            f.write('shape = reader.OneShape()\n\n')
            
            f.write('# Inicializar visualización\n')
            f.write('display, start_display, _, _ = init_display()\n\n')
            
            f.write('# Mostrar modelo\n')
            f.write('ais_shape = AIS_Shape(shape)\n')
            f.write('display.Context.Display(ais_shape, True)\n')
            f.write('display.Context.SetTransparency(ais_shape, 0.7, True)\n')
            f.write('display.Context.SetColor(ais_shape, Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB), True)\n\n')
            
            f.write('# Colores para heat stakes válidos\n')
            f.write('colors = [\n')
            f.write('    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),\n')
            f.write('    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),\n')
            f.write(']\n\n')
            
            f.write('# Mostrar heat stakes VÁLIDOS (colores)\n')
            f.write('for i, (x, y, z, conf) in enumerate(heat_stakes):\n')
            f.write('    # Tamaño según confianza\n')
            f.write('    if conf == "HIGH":\n')
            f.write('        radius = 8.0 * 1.2\n')
            f.write('    elif conf == "MEDIUM":\n')
            f.write('        radius = 8.0\n')
            f.write('    else:\n')
            f.write('        radius = 8.0 * 0.8\n')
            f.write('    \n')
            f.write('    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()\n')
            f.write('    ais_sphere = AIS_Shape(sphere)\n')
            f.write('    display.Context.Display(ais_sphere, True)\n')
            f.write('    color = colors[i % len(colors)]\n')
            f.write('    display.Context.SetColor(ais_sphere, Quantity_Color(*color, Quantity_TOC_RGB), True)\n')
            f.write('    \n')
            f.write('    # Agregar etiqueta\n')
            f.write('    label = f"HS-{i+1}"\n')
            f.write('    text_pos = gp_Pnt(x, y, z + radius * 1.5)\n')
            f.write('    display.DisplayMessage(text_pos, label, height=radius * 0.8, message_color=(0, 0, 0))\n\n')
            
            f.write('# Mostrar clusters RECHAZADOS (negro)\n')
            f.write('for j, (x, y, z) in enumerate(rejected_clusters):\n')
            f.write('    radius = 8.0 * 0.6\n')
            f.write('    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()\n')
            f.write('    ais_sphere = AIS_Shape(sphere)\n')
            f.write('    display.Context.Display(ais_sphere, True)\n')
            f.write('    display.Context.SetColor(ais_sphere, Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB), True)\n')
            f.write('    \n')
            f.write('    # Agregar etiqueta\n')
            f.write('    label = f"R-{j+1}"\n')
            f.write('    text_pos = gp_Pnt(x, y, z + radius * 1.5)\n')
            f.write('    display.DisplayMessage(text_pos, label, height=radius * 0.8, message_color=(0.3, 0.3, 0.3))\n\n')
            
            f.write('print(f"Visualizando {len(heat_stakes)} heat stakes válidos")\n')
            f.write('if rejected_clusters:\n')
            f.write('    print(f"            + {len(rejected_clusters)} clusters rechazados (negro)")\n')
            f.write('display.FitAll()\n')
            f.write('start_display()\n')
        
        print(f"✓ Script creado: {output_file}")
        print(f"   Para visualizar más tarde, ejecuta: python {output_file}")


def main():
    """Función principal"""
    
    if len(sys.argv) < 2:
        print("="*80)
        print("DETECTOR DE HEAT STAKES - Uso")
        print("="*80)
        print("\n📝 Sintaxis:")
        print("   python detector.py <archivo.step> [eps] [min_samples] [--all] [--view] [--show-rejected] [--export-view] [--strict]")
        print("\n📌 Ejemplos:")
        print("   python detector.py modelo.step")
        print("   python detector.py modelo.step --view --show-rejected")
        print("   python detector.py modelo.step 25.0 4 --all --view --show-rejected")
        print("   python detector.py modelo.step --strict")
        print("\n⚙️  Parámetros:")
        print("   eps              Distancia máxima entre cilindros (default: 25.0mm)")
        print("   min_samples      Mínimo cilindros por grupo (default: 5)")
        print("   --all            Mostrar también clusters rechazados en terminal")
        print("   --view           Abrir visualización 3D interactiva")
        print("   --show-rejected  Mostrar clusters rechazados en NEGRO en 3D")
        print("   --export-view    Crear script Python para visualizar después")
        print("   --strict         Modo estricto: solo acepta 7±2 cilindros")
        print("\n🎯 Modos de Detección:")
        print("   FLEXIBLE (default): Acepta 5-25 cilindros, confianza por cercanía a 7")
        print("   ESTRICTO (--strict): Solo acepta 7±2 cilindros")
        print("\n🎨 Visualización 3D:")
        print("   🔴🟢🔵🟡 Heat stakes VÁLIDOS (colores variados, tamaño según confianza)")
        print("   ⚫ Clusters RECHAZADOS (negro, pequeños) [con --show-rejected]")
        print("\n💡 Tips:")
        print("   - Usa --show-rejected para ver QUÉ no fue clasificado como heat stake")
        print("   - Usa --all para ver detalles de clusters rechazados en terminal")
        print("   - Detecta menos? → Aumenta eps (30-40mm) o reduce min_samples (3-4)")
        print("   - Detecta más? → Reduce eps (15-20mm) o aumenta min_samples (6-8)")
        print("="*80)
        sys.exit(1)
    
    step_file = sys.argv[1]
    eps = None
    min_samples = None
    show_all = '--all' in sys.argv
    show_view = '--view' in sys.argv
    show_rejected = '--show-rejected' in sys.argv
    export_view_script = '--export-view' in sys.argv
    strict_mode = '--strict' in sys.argv
    
    # Parse numeric arguments
    for i, arg in enumerate(sys.argv[2:], start=2):
        if not arg.startswith('--'):
            try:
                val = float(arg)
                if eps is None:
                    eps = val
                elif min_samples is None:
                    min_samples = int(val)
            except ValueError:
                pass
    
    # Crear y ejecutar detector
    detector = HeatStakeDetector(step_file)
    
    # Configurar modo estricto si se solicitó
    if strict_mode:
        detector.STRICT_MODE = True
        print("\n⚠️  Modo ESTRICTO activado: Solo 7±2 cilindros\n")
    
    heat_stakes = detector.detect_heat_stakes(
        eps=eps, 
        min_samples=min_samples,
        show_all_clusters=show_all
    )
    
    # Exportar coordenadas
    if heat_stakes:
        output_file = 'heat_stakes_coordinates.txt'
        print(f"\n💾 Exportando a '{output_file}'...")
        
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write("COORDENADAS DE HEAT STAKES DETECTADOS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total detectados: {len(heat_stakes)}\n")
            f.write(f"Archivo: {step_file}\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("="*80 + "\n")
            f.write("HEAT STAKES VÁLIDOS\n")
            f.write("="*80 + "\n\n")
            
            for i, hs in enumerate(heat_stakes):
                c = hs['analysis']['centroid']
                n = hs['analysis']['num_cylinders']
                s = hs['analysis']['max_spread']
                conf = hs['validation']['confidence']
                
                f.write(f"Heat Stake HS-{i+1} {'─' * (60 - len(str(i+1)))}\n")
                f.write(f"ID:         HS-{i+1}\n")
                f.write(f"Centroide:  X={c[0]:.4f}, Y={c[1]:.4f}, Z={c[2]:.4f}\n")
                f.write(f"Cilindros:  {n}\n")
                f.write(f"Dispersión: {s:.2f} mm\n")
                f.write(f"Confianza:  {conf}\n")
                f.write(f"{'─' * 70}\n\n")
            
            # CSV format section para fácil importación
            f.write("\n" + "="*80 + "\n")
            f.write("FORMATO CSV (para importar a Excel/software)\n")
            f.write("="*80 + "\n\n")
            f.write("ID,X,Y,Z,Cilindros,Dispersión,Confianza\n")
            for i, hs in enumerate(heat_stakes):
                c = hs['analysis']['centroid']
                n = hs['analysis']['num_cylinders']
                s = hs['analysis']['max_spread']
                conf = hs['validation']['confidence']
                f.write(f"HS-{i+1},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{n},{s:.2f},{conf}\n")
            
            # Sección de rechazados (si existen)
            if detector.rejected_clusters:
                f.write("\n\n" + "="*80 + "\n")
                f.write("CLUSTERS RECHAZADOS (No clasificados como Heat Stakes)\n")
                f.write("="*80 + "\n\n")
                
                for j, item in enumerate(detector.rejected_clusters):
                    c = item['analysis']['centroid']
                    n = item['analysis']['num_cylinders']
                    s = item['analysis']['max_spread']
                    val = item['validation']
                    
                    f.write(f"Cluster Rechazado R-{j+1} {'─' * (50 - len(str(j+1)))}\n")
                    f.write(f"ID:         R-{j+1}\n")
                    f.write(f"Centroide:  X={c[0]:.4f}, Y={c[1]:.4f}, Z={c[2]:.4f}\n")
                    f.write(f"Cilindros:  {n}\n")
                    f.write(f"Dispersión: {s:.2f} mm\n")
                    f.write(f"Razón de rechazo:\n")
                    f.write(f"- Cilindros: {'✓' if val['cylinder_check'] else '✗'} {val['details']['cylinders']}\n")
                    f.write(f"- Dispersión: {'✓' if val['spread_check'] else '✗'} {val['details']['spread']}\n")
                    f.write(f"- Radio: {'✓' if val['radius_check'] else '✗'} {val['details']['avg_radius']}\n")
                    if 'density_check' in val:
                        f.write(f"- Densidad: {'✓' if val['density_check'] else '✗'} {val['details']['density']}\n")
                    f.write(f"{'─' * 70}\n\n")
                
                # CSV para rechazados
                f.write("\nCSV - RECHAZADOS:\n")
                f.write("ID,X,Y,Z,Cilindros,Dispersión,Razón\n")
                for j, item in enumerate(detector.rejected_clusters):
                    c = item['analysis']['centroid']
                    n = item['analysis']['num_cylinders']
                    s = item['analysis']['max_spread']
                    razon = "Criterios_no_cumplidos"
                    f.write(f"R-{j+1},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{n},{s:.2f},{razon}\n")
        
        print(f"✓ Archivo creado")
        
        # Visualización 3D
        if show_view:
            detector.visualize_heat_stakes(marker_size=8.0, show_model=True)


if __name__ == "__main__":
    main()