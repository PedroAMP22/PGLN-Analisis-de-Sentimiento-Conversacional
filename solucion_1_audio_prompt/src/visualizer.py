import networkx as nx
import matplotlib.pyplot as plt
import os

# AÑADIMOS el parámetro 'nombre_salida'
def generar_grafo_visual(datos, output_folder, nombre_salida="grafo_resultado"):
    print(f"🎨 Generando imagen del grafo: {nombre_salida}.png ...")
    
    G = nx.DiGraph()
    
    colores_nodos = []
    labels_nodos = {}
    
    # Procesar Nodos
    for nodo in datos.get("nodos", []):
        G.add_node(nodo['id'])
        colores_nodos.append(nodo.get('color', '#CCCCCC'))
        labels_nodos[nodo['id']] = f"{nodo['id']}\n({nodo['emocion']})"
        
    # Procesar Aristas
    labels_aristas = {}
    for arista in datos.get("aristas", []):
        G.add_edge(arista['origen'], arista['destino'])
        labels_aristas[(arista['origen'], arista['destino'])] = arista['relacion']
        
    # Dibujar
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.9)
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=colores_nodos, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels_nodos, font_size=9, font_weight="bold")
    
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, arrowsize=20, edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_aristas, font_color="red", font_size=8)
    
    plt.title(f"Grafo de Emociones: {nombre_salida}")
    plt.axis('off')
    
    # USAMOS el nombre personalizado
    ruta_salida = os.path.join(output_folder, f"{nombre_salida}.png")
    plt.savefig(ruta_salida)
    plt.close()
    
    print(f"🖼️  Imagen guardada en: {ruta_salida}")