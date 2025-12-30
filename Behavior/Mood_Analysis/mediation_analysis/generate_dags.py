#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAG Generator for Mediation Analysis (Publication Quality)
Final Version: Fixed labels, consistent styling, and clear interaction arrows.
"""

import matplotlib.pyplot as plt
import networkx as nx
import os

# Output directory for DAGs (relative to project root)
OUTPUT_DIR = "results/Behavior/mediation_analysis/DAGs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de Estilo Unificado
NODE_SIZE = 5500
FONT_SIZE = 9
FONT_WEIGHT = "bold"
EDGE_WIDTH = 2
ARROW_SIZE = 25

# Colores estandarizados para todos los gráficos
COLORS = {
    "X": "#e3f2fd", # Azul claro (Input)
    "M": "#fff9c4", # Amarillo (Mediator)
    "Y": "#ffebee", # Rosado (Outcome)
    "Cov": "#f5f5f5", # Gris muy claro (Covariables)
    "Mod": "#e0e0e0"  # Gris medio (Moderador)
}

def draw_dag(model_type="forward"):
    """Dibuja los DAGs simples (Forward/Reverse) con etiquetas visibles."""
    G = nx.DiGraph()
    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    
    if model_type == "forward":
        title = "Forward Mediation: Mood as Mechanism"
        pos = {
            "Group": (0, 0), 
            "Mood\n(Mediator)": (1, 1),
            "Thoughts\n(Outcome)": (2, 0), 
            "Time-on-Task": (1, -0.8)
        }
        # Definir colores por nodo
        n_colors = [COLORS["X"], COLORS["M"], COLORS["Y"], COLORS["Cov"]]
        
        edges = [("Group", "Mood\n(Mediator)"), ("Mood\n(Mediator)", "Thoughts\n(Outcome)"),
                 ("Group", "Thoughts\n(Outcome)"), ("Time-on-Task", "Mood\n(Mediator)"),
                 ("Time-on-Task", "Thoughts\n(Outcome)")]
        edge_labels = {("Group", "Mood\n(Mediator)"): "a", ("Mood\n(Mediator)", "Thoughts\n(Outcome)"): "b",
                       ("Group", "Thoughts\n(Outcome)"): "c'"}

    elif model_type == "reverse":
        title = "Reverse Mediation: Thoughts driving Mood Change"
        pos = {
            "Group": (0, 0), 
            "Thoughts\n(Mediator)": (1, 1),
            "Mood Post\n(Outcome)": (2, 0), 
            "Mood Pre\n(Baseline)": (0.5, -0.8), 
            "Time": (1.5, -0.8)
        }
        n_colors = [COLORS["X"], COLORS["M"], COLORS["Y"], COLORS["Cov"], COLORS["Cov"]]
        
        edges = [("Group", "Thoughts\n(Mediator)"), ("Thoughts\n(Mediator)", "Mood Post\n(Outcome)"),
                 ("Group", "Mood Post\n(Outcome)"), ("Mood Pre\n(Baseline)", "Thoughts\n(Mediator)"),
                 ("Mood Pre\n(Baseline)", "Mood Post\n(Outcome)"), ("Time", "Thoughts\n(Mediator)"),
                 ("Time", "Mood Post\n(Outcome)")]
        edge_labels = {("Group", "Thoughts\n(Mediator)"): "a", ("Thoughts\n(Mediator)", "Mood Post\n(Outcome)"): "b",
                       ("Group", "Mood Post\n(Outcome)"): "c'"}

    # Dibujar Nodos
    nx.draw_networkx_nodes(G, pos, nodelist=pos.keys(), node_size=NODE_SIZE, 
                           node_color=n_colors, edgecolors="black", linewidths=1.5)
    
    # Dibujar Etiquetas (FORCE Z-ORDER)
    nx.draw_networkx_labels(G, pos, labels={k:k for k in pos.keys()}, 
                            font_size=FONT_SIZE, font_weight=FONT_WEIGHT, font_family="sans-serif")
    
    # Dibujar Aristas
    # Separar principales de covariables
    cov_terms = ["Time", "Baseline", "Time-on-Task"]
    main_edges = [e for e in edges if not any(x in e[0] for x in cov_terms)]
    cov_edges = [e for e in edges if any(x in e[0] for x in cov_terms)]
    
    nx.draw_networkx_edges(G, pos, edgelist=main_edges, width=EDGE_WIDTH, arrowstyle='-|>', 
                           arrowsize=ARROW_SIZE, edge_color="black", node_size=NODE_SIZE)
    nx.draw_networkx_edges(G, pos, edgelist=cov_edges, width=1.5, arrowstyle='-|>', 
                           arrowsize=15, edge_color="gray", style="dashed", node_size=NODE_SIZE)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color="red", rotate=False, label_pos=0.5)

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, f"DAG_{model_type}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"DAG saved: {out_path}")
    plt.close()

def draw_cyberball_dag():
    """
    Dibuja el DAG de Cyberball con etiquetas forzadas y estilo consistente.
    """
    G = nx.DiGraph()
    
    # 1. Posiciones (Diseño espacioso)
    pos = {
        "Cyberball\n(Condition)": (0, 0),      # Izquierda
        "Mood Change\n(Mediator)": (2, 2),     # Centro Arriba
        "Thoughts\n(Valence)": (4, 0),         # Derecha
        "Group\n(Moderator)": (-1.0, 2.5),     # Arriba Izquierda (fuera del camino)
        "Baseline Mood": (2, 3.8)              # Muy arriba
    }
    
    # Nodos auxiliares para flechas especiales
    pos["_mod_target"] = (1, 1) # Punto medio del path A
    
    # 2. Lienzo y Colores
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_xlim(-2.0, 5.0)
    ax.set_ylim(-1.0, 4.5)
    
    node_list = ["Cyberball\n(Condition)", "Mood Change\n(Mediator)", "Thoughts\n(Valence)", "Group\n(Moderator)", "Baseline Mood"]
    color_map = [COLORS["X"], COLORS["M"], COLORS["Y"], COLORS["Mod"], COLORS["Cov"]]
    
    # 3. Dibujar Nodos
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=NODE_SIZE, 
                           node_color=color_map, edgecolors="black", linewidths=1.5)
    
    # 4. Dibujar Etiquetas (CRÍTICO: zorder alto para que se vea)
    nx.draw_networkx_labels(G, pos, labels={k:k for k in node_list}, 
                            font_size=FONT_SIZE, font_weight=FONT_WEIGHT, font_family="sans-serif")
    
    # 5. Dibujar Flechas
    # Estilo común
    arrow_args = dict(lw=EDGE_WIDTH, color="black", mutation_scale=ARROW_SIZE, shrinkA=30, shrinkB=30)
    
    # a, b, c'
    ax.annotate("", xy=pos["Mood Change\n(Mediator)"], xytext=pos["Cyberball\n(Condition)"], arrowprops=dict(arrowstyle="-|>", **arrow_args))
    ax.annotate("", xy=pos["Thoughts\n(Valence)"], xytext=pos["Mood Change\n(Mediator)"], arrowprops=dict(arrowstyle="-|>", **arrow_args))
    ax.annotate("", xy=pos["Thoughts\n(Valence)"], xytext=pos["Cyberball\n(Condition)"], arrowprops=dict(arrowstyle="-|>", **arrow_args))
    
    # Moderación (Roja y Curva)
    ax.annotate("", xy=pos["_mod_target"], xytext=pos["Group\n(Moderator)"], 
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#d32f2f", ls="--", 
                                connectionstyle="arc3,rad=0.2", mutation_scale=20, shrinkA=30))
    
    # Control (Gris Punteada)
    ax.annotate("", xy=pos["Mood Change\n(Mediator)"], xytext=pos["Baseline Mood"], 
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="gray", ls=":", mutation_scale=15, shrinkA=30, shrinkB=30))
    
    # 6. Etiquetas de los Caminos (Texto suelto)
    plt.text(0.8, 1.4, "a\n(Interaction)", color="#d32f2f", fontsize=11, fontweight="bold", ha="center", va="center", zorder=15, backgroundcolor="white")
    plt.text(3.0, 1.2, "b", color="black", fontsize=12, fontweight="bold", ha="center", va="center", zorder=15)
    plt.text(2.0, -0.2, "c'", color="black", fontsize=12, fontweight="bold", ha="center", va="center", zorder=15)
    
    plt.title("Moderated Mediation: Cyberball Impact\n(Does Risk Group react more intensely to exclusion?)", fontsize=16, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "DAG_cyberball_moderated.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"DAG saved: {out_path}")
    plt.close()


def draw_cyberball_delayed_mood_dag():
    """
    Draw DAG for Delayed Mood Mediation:
    Cyberball → Thoughts (SART) → Delayed Mood (after SART)
    Tests whether thoughts during SART mediate the delayed effect on mood.
    """
    G = nx.DiGraph()
    
    # 1. Positions (Spacious design)
    pos = {
        "Cyberball\n(Condition)": (0, 0),           # Left
        "Thoughts\n(Mediator)": (2, 2),             # Center Top
        "Delayed Mood\n(Outcome)": (4, 0),          # Right
        "Group\n(Moderator)": (-1.0, 2.5),          # Top Left (out of the way)
        "Baseline Mood": (2, 3.8)                   # Very top
    }
    
    # Auxiliary nodes for special arrows
    pos["_mod_target"] = (1, 1)  # Midpoint of path A
    
    # 2. Canvas and Colors
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_xlim(-2.0, 5.0)
    ax.set_ylim(-1.0, 4.5)
    
    node_list = ["Cyberball\n(Condition)", "Thoughts\n(Mediator)", "Delayed Mood\n(Outcome)", "Group\n(Moderator)", "Baseline Mood"]
    color_map = [COLORS["X"], COLORS["M"], COLORS["Y"], COLORS["Mod"], COLORS["Cov"]]
    
    # 3. Draw Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=NODE_SIZE, 
                           node_color=color_map, edgecolors="black", linewidths=1.5)
    
    # 4. Draw Labels
    nx.draw_networkx_labels(G, pos, labels={k:k for k in node_list}, 
                            font_size=FONT_SIZE, font_weight=FONT_WEIGHT, font_family="sans-serif")
    
    # 5. Draw Arrows
    arrow_args = dict(lw=EDGE_WIDTH, color="black", mutation_scale=ARROW_SIZE, shrinkA=30, shrinkB=30)
    
    # a, b, c'
    ax.annotate("", xy=pos["Thoughts\n(Mediator)"], xytext=pos["Cyberball\n(Condition)"], arrowprops=dict(arrowstyle="-|>", **arrow_args))
    ax.annotate("", xy=pos["Delayed Mood\n(Outcome)"], xytext=pos["Thoughts\n(Mediator)"], arrowprops=dict(arrowstyle="-|>", **arrow_args))
    ax.annotate("", xy=pos["Delayed Mood\n(Outcome)"], xytext=pos["Cyberball\n(Condition)"], arrowprops=dict(arrowstyle="-|>", **arrow_args))
    
    # Moderation (Red and Curved)
    ax.annotate("", xy=pos["_mod_target"], xytext=pos["Group\n(Moderator)"], 
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#d32f2f", ls="--", 
                                connectionstyle="arc3,rad=0.2", mutation_scale=20, shrinkA=30))
    
    # Control (Gray Dotted)
    ax.annotate("", xy=pos["Thoughts\n(Mediator)"], xytext=pos["Baseline Mood"], 
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="gray", ls=":", mutation_scale=15, shrinkA=30, shrinkB=30))
    ax.annotate("", xy=pos["Delayed Mood\n(Outcome)"], xytext=pos["Baseline Mood"], 
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="gray", ls=":", mutation_scale=15, shrinkA=30, shrinkB=30))
    
    # 6. Path Labels (Text)
    plt.text(0.8, 1.4, "a\n(Interaction)", color="#d32f2f", fontsize=11, fontweight="bold", ha="center", va="center", zorder=15, backgroundcolor="white")
    plt.text(3.0, 1.2, "b", color="black", fontsize=12, fontweight="bold", ha="center", va="center", zorder=15)
    plt.text(2.0, -0.2, "c'", color="black", fontsize=12, fontweight="bold", ha="center", va="center", zorder=15)
    
    plt.title("Delayed Mood Mediation: Cyberball → Thoughts → Mood\n(Do thoughts during SART mediate delayed mood changes?)", fontsize=16, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "DAG_cyberball_delayed_mood.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"DAG saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    draw_dag("forward")
    draw_dag("reverse")
    draw_cyberball_dag()
    draw_cyberball_delayed_mood_dag()