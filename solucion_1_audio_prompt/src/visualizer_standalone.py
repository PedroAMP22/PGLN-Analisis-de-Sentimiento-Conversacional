import os
import json
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio

#configuracion de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

#estilo visual para el gif
NODE_ACTIVE_BG = '#FFFFFF'
NODE_INACTIVE_BG = '#F5F5F5'
NODE_BORDER = '#000000'

def generar_video_evolutivo(ruta_json):
    print(f"Generando video para: {os.path.basename(ruta_json)}...")
    
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    timeline = datos.get("timeline", [])
    if not timeline: return print("JSON sin timeline.")

    #ordenacion y censo
    todos_personajes = set()
    for fase in timeline:
        for n in fase.get("nodos", []): todos_personajes.add(n['id'])
        for a in fase.get("aristas", []):
            todos_personajes.add(a['origen'])
            todos_personajes.add(a['destino'])
            
    if not todos_personajes: return print("No hay personajes.")

    lista_personajes = sorted(list(todos_personajes))

    #posicines fijas
    G_global = nx.DiGraph()
    G_global.add_nodes_from(lista_personajes)
    pos_fija = nx.circular_layout(G_global, scale=0.85) 

    #carpeta temporal de frames
    temp_dir = os.path.join(OUTPUT_DIR, "temp_frames")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    filenames = []

    #generamos los frames
    for i, fase in enumerate(timeline):
        plt.figure(figsize=(10, 10))
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        
        G_frame = nx.DiGraph()
        G_frame.add_nodes_from(lista_personajes)
        
        # A. NODOS
        colores_fondo = []
        labels_nodos = {}
        activos = {n['id']: n for n in fase.get("nodos", [])}
        
        for p in lista_personajes:
            if p in activos:
                colores_fondo.append(NODE_ACTIVE_BG)
                emocion = activos[p].get('emocion', '').upper()
                labels_nodos[p] = f"{p}\n[{emocion}]"
            else:
                colores_fondo.append(NODE_INACTIVE_BG)
                labels_nodos[p] = p

        nx.draw_networkx_nodes(
            G_frame, pos_fija, node_size=4800, 
            node_color=colores_fondo, edgecolors=NODE_BORDER, linewidths=1.5
        )
        nx.draw_networkx_labels(G_frame, pos_fija, labels=labels_nodos, font_size=8, font_weight="bold")

        # B. ARISTAS (Lógica de agrupación <->)
        interacciones = {}
        for a in fase.get("aristas", []):
            u, v = a['origen'], a['destino']
            if u in lista_personajes and v in lista_personajes:
                pair = frozenset([u, v])
                if pair not in interacciones: interacciones[pair] = []
                interacciones[pair].append(a)

        aristas_uni = []
        labels_uni = {}
        aristas_bi = []
        labels_bi = {}

        for pair, lista in interacciones.items():
            u, v = list(pair)[0], list(pair)[-1]
            
            #unidireccional
            if len(lista) == 1 or (len(lista) > 1 and all(x['origen'] == lista[0]['origen'] for x in lista)):
                item = lista[0]
                aristas_uni.append((item['origen'], item['destino']))
                labels_uni[(item['origen'], item['destino'])] = item['relacion']
            #bidireccional
            else:
                rel_u_v = next((x['relacion'] for x in lista if x['origen'] == u), "")
                rel_v_u = next((x['relacion'] for x in lista if x['origen'] == v), "")
                aristas_bi.append((u, v))
                labels_bi[(u, v)] = f"{rel_u_v} / {rel_v_u}".strip(" / ")

        #dibujamos unidirecionales
        if aristas_uni:
            nx.draw_networkx_edges(
                G_frame, pos_fija, edgelist=aristas_uni, width=1.5,
                edge_color='black', arrowsize=25, arrowstyle='-|>',
                node_size=4800, connectionstyle="arc3,rad=0"
            )
            nx.draw_networkx_edge_labels(
                G_frame, pos_fija, edge_labels=labels_uni, font_size=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5)
            )

        #dibujamos Bidireccionales
        if aristas_bi:
            nx.draw_networkx_edges(
                G_frame, pos_fija, edgelist=aristas_bi, width=1.5,
                edge_color='black', arrowsize=25, arrowstyle='<|-|>',
                node_size=4800, connectionstyle="arc3,rad=0"
            )
            nx.draw_networkx_edge_labels(
                G_frame, pos_fija, edge_labels=labels_bi, font_size=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5)
            )

        plt.axis('off')
        filename = os.path.join(temp_dir, f"frame_{i}.png")
        plt.savefig(filename, dpi=100)
        plt.close()
        filenames.append(filename)

    #generamos el gif
    if filenames:
        nombre_gif = os.path.basename(ruta_json).replace('.json', '.gif')
        ruta_gif = os.path.join(OUTPUT_DIR, nombre_gif)
        print("Renderizando GIF...")
        with imageio.get_writer(ruta_gif, mode='I', duration=3000, loop=0) as writer:
            for filename in filenames:
                writer.append_data(imageio.imread(filename))
        shutil.rmtree(temp_dir)
        print(f"Video guardado en: {ruta_gif}")

def main():
    #seleccionamos el json mediante el cual queremso generar el gif
    archivos = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    if not archivos: return print("No hay JSONs en output/")
    
    print("\n--- SELECCIONA JSON PARA VISUALIZAR ---")
    for i, f in enumerate(archivos): print(f"[{i+1}] {f}")
    
    try:
        idx = int(input("Número: ")) - 1
        generar_video_evolutivo(os.path.join(OUTPUT_DIR, archivos[idx]))
    except ValueError:
        print("Número inválido")

if __name__ == "__main__":
    main()