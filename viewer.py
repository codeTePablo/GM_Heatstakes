"""
DETECTOR DE HEAT STAKES - Clustering de Cilindros (Optimizado para piezas ensambladas)
Detecta heat stakes en archivos STEP usando clustering de cilindros.

------------------------------------------------------------
Developer: Pablo
Contacto:  codewithpablo@gmail.com
------------------------------------------------------------
"""

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shape
import numpy as np

class STEPReader:
    """Clase para leer y analizar archivos STEP"""
    
    def __init__(self, filepath):
        """
        Inicializa el lector de STEP
        
        Args:
            filepath (str): Ruta al archivo STEP
        """
        self.filepath = filepath
        self.shape = None
        self.reader = STEPControl_Reader()
        self.display = None
        self.start_display = None
        self.add_menu = None
        self.add_function_to_menu = None
        
    def load_step_file(self):
        """
        Carga el archivo STEP y retorna el shape principal
        
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        try:
            # Leer el archivo STEP
            status = self.reader.ReadFile(self.filepath)
            
            if status != IFSelect_RetDone:
                print(f"Error: No se pudo leer el archivo {self.filepath}")
                return False
            
            # Transferir las formas del archivo
            self.reader.TransferRoots()
            self.shape = self.reader.OneShape()
            
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def get_basic_info(self):
        """
        Obtiene información básica de la geometría
        
        Returns:
            dict: Diccionario con información básica
        """
        if self.shape is None:
            print("Error: Primero debe cargar un archivo STEP")
            return None
        
        topo = TopologyExplorer(self.shape)
        
        info = {
            'num_solids': topo.number_of_solids(),
            'num_faces': topo.number_of_faces(),
            'num_edges': topo.number_of_edges(),
            'num_vertices': topo.number_of_vertices()
        }
        
        return info
    
    def calculate_center_of_gravity(self):
        """
        Calcula el centro de gravedad del shape completo
        
        Returns:
            tuple: Coordenadas (x, y, z) del centro de gravedad
        """
        if self.shape is None:
            print("Error: Primero debe cargar un archivo STEP")
            return None
        
        # Calcular propiedades volumétricas
        props = GProp_GProps()
        brepgprop_VolumeProperties(self.shape, props)
        
        # Obtener centro de gravedad
        cog = props.CentreOfMass()
        
        return (cog.X(), cog.Y(), cog.Z())
    
    def get_all_solids(self):
        """
        Extrae todos los sólidos de la geometría
        
        Returns:
            list: Lista de sólidos encontrados
        """
        if self.shape is None:
            print("Error: Primero debe cargar un archivo STEP")
            return []
        
        solids = []
        explorer = TopExp_Explorer(self.shape, TopAbs_SOLID)
        
        while explorer.More():
            solid = explorer.Current()
            solids.append(solid)
            explorer.Next()
        
        return solids
    
    def print_summary(self):
        """Imprime un resumen de la geometría cargada"""
        if self.shape is None:
            print("No hay geometría cargada")
            return
        
        info = self.get_basic_info()
        cog = self.calculate_center_of_gravity()
        
        print("\n" + "="*50)
        print("RESUMEN DE GEOMETRÍA")
        print("="*50)
        print(f"Archivo: {self.filepath}")
        print(f"\nElementos topológicos:")
        print(f"  - Sólidos: {info['num_solids']}")
        print(f"  - Caras: {info['num_faces']}")
        print(f"  - Aristas: {info['num_edges']}")
        print(f"  - Vértices: {info['num_vertices']}")
        print(f"\nCentro de gravedad:")
        print(f"  X: {cog[0]:.3f} mm")
        print(f"  Y: {cog[1]:.3f} mm")
        print(f"  Z: {cog[2]:.3f} mm")
        print("="*50 + "\n")


    def visualize(self, color=(0.7, 0.7, 0.8), transparency=0.0):
        """
        Visualiza la geometría en una ventana 3D interactiva
        
        Args:
            color (tuple): Color RGB normalizado (0-1, 0-1, 0-1)
            transparency (float): Transparencia (0=opaco, 1=transparente)
        """
        if self.shape is None:
            print("Error: Primero debe cargar un archivo STEP")
            return
        
        print("\n🔍 Iniciando visualizador 3D...")
        print("Controles:")
        print("-" * 50)
        
        try:
            # Inicializar el display
            self.display, self.start_display, self.add_menu, self.add_function_to_menu = init_display()
            
            # Configurar color
            r, g, b = color
            shape_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
            
            # Mostrar la geometría
            ais_shape = self.display.DisplayShape(
                self.shape,
                color=shape_color,
                transparency=transparency,
                update=True
            )[0]
            
            # Configurar vista
            self.display.FitAll()
            self.display.View.SetBackgroundColor(Quantity_Color(0.95, 0.95, 0.95, Quantity_TOC_RGB))
            
            # Mostrar ejes y grid (opcional)
            # self.display.display_triedron()
            
            # Iniciar el loop de visualización
            self.start_display()
            
        except Exception as e:
            print(f"Error al visualizar: {e}")
    
    def visualize_with_solids_colored(self):
        """
        Visualiza la geometría con cada sólido en un color diferente
        Útil para identificar componentes individuales
        """
        if self.shape is None:
            print("Error: Primero debe cargar un archivo STEP")
            return
        
        print("\n🎨 Iniciando visualizador con sólidos coloreados...")
        
        try:
            # Inicializar el display
            self.display, self.start_display, self.add_menu, self.add_function_to_menu = init_display()
            
            # Obtener todos los sólidos
            solids = self.get_all_solids()
            
            # Paleta de colores
            colors = [
                (1.0, 0.0, 0.0),  # Rojo
                (0.0, 1.0, 0.0),  # Verde
                (0.0, 0.0, 1.0),  # Azul
                (1.0, 1.0, 0.0),  # Amarillo
                (1.0, 0.0, 1.0),  # Magenta
                (0.0, 1.0, 1.0),  # Cian
                (1.0, 0.5, 0.0),  # Naranja
                (0.5, 0.0, 1.0),  # Púrpura
                (0.0, 0.5, 0.5),  # Verde azulado
                (0.5, 0.5, 0.0),  # Oliva
            ]
            
            # Mostrar cada sólido con un color diferente
            for i, solid in enumerate(solids):
                color = colors[i % len(colors)]
                r, g, b = color
                shape_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
                
                self.display.DisplayShape(
                    solid,
                    color=shape_color,
                    transparency=0.0,
                    update=False
                )
            
            # Actualizar display
            self.display.FitAll()
            self.display.View.SetBackgroundColor(Quantity_Color(0.95, 0.95, 0.95, Quantity_TOC_RGB))
            
            print(f"✓ Visualizando {len(solids)} sólido(s) con colores diferentes\n")
            print("-" * 50)
            
            # Iniciar el loop de visualización
            self.start_display()
            
        except Exception as e:
            print(f"Error al visualizar: {e}")


def main():
    """Función principal para ejecutar el lector de STEP"""
    
    # Nombre del archivo
    filename = "PKG - COMPONENTES CON HEAT STAKES - UAEMEX/PKG - COMPONENTES CON HEAT STAKES - UAEMEX/EXERCISE 1 -DOOR PANELS.stp"
    
    # Crear instancia del lector
    reader = STEPReader(filename)
    
    # Cargar el archivo
    if reader.load_step_file():
        # Mostrar resumen
        reader.print_summary()
        
        # Obtener información adicional
        info = reader.get_basic_info()
        solids = reader.get_all_solids()
        
        print(f"Se encontraron {len(solids)} sólido(s) en la geometría")
        
        # Preguntar si desea visualizar
        print("\n¿Desea visualizar la geometría?")
        print("1. Visualizar con color único")
        print("2. Visualizar con sólidos coloreados (recomendado)")
        print("3. No visualizar")
        
        try:
            opcion = input("\nSeleccione una opción (1/2/3): ").strip()
            
            if opcion == "1":
                reader.visualize(color=(0.2, 0.6, 0.9), transparency=0.0)
            elif opcion == "2":
                reader.visualize_with_solids_colored()
            else:
                print("\nVisualizador no iniciado. Análisis completado.")
        except:
            print("\nNo se pudo leer la opción. Omitiendo visualización.")
        
        return reader
    else:
        print("No se pudo cargar el archivo STEP")
        return None


if __name__ == "__main__":
    # Ejecutar el programa
    step_reader = main()
    
    # El objeto step_reader contiene toda la geometría
    # y puede ser usado para análisis posteriores
    if step_reader is not None:
        print("\n✓ Objeto 'step_reader' disponible para análisis adicionales")
        print("\nPara visualizar posteriormente, use:")
        print("  step_reader.visualize()  # Color único")
        print("  step_reader.visualize_with_solids_colored()  # Sólidos coloreados")