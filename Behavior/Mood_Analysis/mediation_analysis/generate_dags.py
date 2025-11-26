#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAG Generator for Mediation Analysis (Fixed Arrows)
Generates publication-quality Directed Acyclic Graphs for Forward and Reverse models.
"""

import matplotlib.pyplot as plt
import networkx as nx
import os

# Directorio para guardar los plots
OUTPUT_DIR = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/mediation_analysis/DAGs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_dag(model_type="forward"):
    """
    Dibuja un DAG estilizado basado en el tipo de modelo.
    """
    G = nx.DiGraph()
    
    # Configuración según el modelo
    if model_type == "forward":
        title = "Forward Mediation: Mood as Mechanism"
        nodes = ["Group", "Mood\n(Mediator)", "Thoughts\n(Outcome)", "Time-on-Task"]
        pos = {
            "Group": (0, 0),
            "Mood\n(Mediator)": (1, 1),
            "Thoughts\n(Outcome)": (2, 0),
            "Time-on-Task": (1, -0.8)
        }
        edges = [
            ("Group", "Mood\n(Mediator)"),
            ("Mood\n(Mediator)", "Thoughts\n(Outcome)"),
            ("Group", "Thoughts\n(Outcome)"),
            ("Time-on-Task", "Mood\n(Mediator)"),
            ("Time-on-Task", "Thoughts\n(Outcome)")
        ]
        edge_labels = {
            ("Group", "Mood\n(Mediator)"): "a",
            ("Mood\n(Mediator)", "Thoughts\n(Outcome)"): "b",
            ("Group", "Thoughts\n(Outcome)"): "c'"
        }
        node_colors = ["#e3f2fd", "#fff9c4", "#ffebee", "#f5f5f5"]

    elif model_type == "reverse":
        title = "Reverse Mediation: Thoughts driving Mood Change"
        nodes = ["Group", "Thoughts\n(Mediator)", "Mood Post\n(Outcome)", "Mood Pre\n(Baseline)", "Time"]
        pos = {
            "Group": (0, 0),
            "Thoughts\n(Mediator)": (1, 1),
            "Mood Post\n(Outcome)": (2, 0),
            "Mood Pre\n(Baseline)": (0.5, -0.8),
            "Time": (1.5, -0.8)
        }
        edges = [
            ("Group", "Thoughts\n(Mediator)"),
            ("Thoughts\n(Mediator)", "Mood Post\n(Outcome)"),
            ("Group", "Mood Post\n(Outcome)"),
            ("Mood Pre\n(Baseline)", "Thoughts\n(Mediator)"),
            ("Mood Pre\n(Baseline)", "Mood Post\n(Outcome)"),
            ("Time", "Thoughts\n(Mediator)"),
            ("Time", "Mood Post\n(Outcome)")
        ]
        edge_labels = {
            ("Group", "Thoughts\n(Mediator)"): "a",
            ("Thoughts\n(Mediator)", "Mood Post\n(Outcome)"): "b",
            ("Group", "Mood Post\n(Outcome)"): "c'"
        }
        node_colors = ["#e3f2fd", "#ffebee", "#fff9c4", "#f5f5f5", "#f5f5f5"]

    # Crear Grafo
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # 1. Dibujar Nodos
    nx.draw_networkx_nodes(G, pos, node_size=5000, node_color=node_colors, edgecolors="black", linewidths=1.5)
    
    # 2. Dibujar Etiquetas de Nodos
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", font_family="sans-serif")
    
    # 3. Dibujar Aristas (CORREGIDO: AÑADIDO node_size y arrowstyle)
    main_edges = [e for e in edges if "Time" not in e[0] and "Baseline" not in e[0]]
    cov_edges = [e for e in edges if "Time" in e[0] or "Baseline" in e[0]]
    
    # Aristas Principales (Sólidas con Flecha Clara)
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=main_edges, 
        width=2, 
        arrowstyle='-|>',   # Fuerza el triángulo cerrado
        arrowsize=20, 
        edge_color="black",
        node_size=5000      # IMPORTANTE: Le dice a la flecha que pare antes de entrar al nodo
    )
    
    # Aristas Covariables (Punteadas con Flecha Clara)
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=cov_edges, 
        width=1.5, 
        arrowstyle='-|>',   # Fuerza el triángulo cerrado
        arrowsize=15, 
        edge_color="gray", 
        style="dashed",
        node_size=5000      # IMPORTANTE: Le dice a la flecha que pare antes de entrar al nodo
    )
    
    # 4. Etiquetas de los Paths
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color="red", rotate=False)

    plt.title(title, fontsize=15, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    # Guardar
    out_path = os.path.join(OUTPUT_DIR, f"DAG_{model_type}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"DAG saved: {out_path}")
    plt.close()

def draw_cyberball_dag():
    """
    Draw a DAG for the Cyberball Moderated Mediation model.
    
    Model: Condition → Mood → Thoughts (Group moderates Path a)
    
    The red dashed arrow from Group to Path a represents the moderation:
    "Exclusion affects mood in everyone, but the Risk Group has an amplifier
    on that connection."
    """
    G = nx.DiGraph()
    
    # Node positions
    pos = {
        "Cyberball\n(Inclusion vs Exclusion)": (0, 1),
        "Mood Change\n(Mediator)": (1.5, 2),
        "Thoughts\n(Valence)": (3, 1),
        "Group\n(Moderator)": (1.5, 0),
        "Baseline Mood": (1.5, 3),
    }
    
    # Invisible nodes for moderation arrows
    pos["_mod_a"] = (0.75, 1.5)
    pos["_mod_c"] = (1.5, 1.05)
    
    # Node colors
    node_colors = ["#e3f2fd", "#fff9c4", "#ffebee", "#e0e0e0", "#f5f5f5"]
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # 1. Draw Real Nodes
    real_nodes = [n for n in pos if not n.startswith("_")]
    nx.draw_networkx_nodes(
        G, pos, nodelist=real_nodes, node_size=4500,
        node_color=node_colors, edgecolors="black", linewidths=1.5
    )
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight="bold", font_family="sans-serif"
    )
    
    # 2. Mediation Paths (Solid Arrows)
    # Path a: Cyberball -> Mood
    ax.annotate(
        "", xy=pos["Mood Change\n(Mediator)"],
        xytext=pos["Cyberball\n(Inclusion vs Exclusion)"],
        arrowprops=dict(
            arrowstyle="-|>", lw=2, color="black",
            mutation_scale=20, shrinkA=25, shrinkB=25
        )
    )
    # Path b: Mood -> Thoughts
    ax.annotate(
        "", xy=pos["Thoughts\n(Valence)"],
        xytext=pos["Mood Change\n(Mediator)"],
        arrowprops=dict(
            arrowstyle="-|>", lw=2, color="black",
            mutation_scale=20, shrinkA=25, shrinkB=25
        )
    )
    # Path c': Cyberball -> Thoughts (direct)
    ax.annotate(
        "", xy=pos["Thoughts\n(Valence)"],
        xytext=pos["Cyberball\n(Inclusion vs Exclusion)"],
        arrowprops=dict(
            arrowstyle="-|>", lw=2, color="black",
            mutation_scale=20, shrinkA=25, shrinkB=25
        )
    )
    
    # 3. Moderation Arrows (Group -> Path a and c')
    # Group moderating Path a (Emotional Reactivity)
    ax.annotate(
        "", xy=pos["_mod_a"], xytext=pos["Group\n(Moderator)"],
        arrowprops=dict(
            arrowstyle="-|>", lw=2, color="#d32f2f", ls="--",
            mutation_scale=15, shrinkA=25
        )
    )
    # Group moderating Path c' (Direct Cognitive Bias)
    ax.annotate(
        "", xy=pos["_mod_c"], xytext=pos["Group\n(Moderator)"],
        arrowprops=dict(
            arrowstyle="-|>", lw=1.5, color="#d32f2f", ls="--",
            mutation_scale=15, shrinkA=25
        )
    )
    
    # 4. Control (Baseline Mood -> Mood Change)
    ax.annotate(
        "", xy=pos["Mood Change\n(Mediator)"], xytext=pos["Baseline Mood"],
        arrowprops=dict(
            arrowstyle="-|>", lw=1.5, color="gray", ls=":",
            mutation_scale=15, shrinkA=25, shrinkB=25
        )
    )
    
    # Path Labels
    plt.text(0.6, 1.6, "a\n(Interaction)", color="red", fontsize=11,
             fontweight="bold", ha="center")
    plt.text(2.4, 1.6, "b", color="black", fontsize=11,
             fontweight="bold", ha="center")
    plt.text(1.5, 0.9, "c'", color="black", fontsize=11,
             fontweight="bold", ha="center")
    
    plt.title(
        "Moderated Mediation: Cyberball Impact\n"
        "(Does Risk Group react more intensely to exclusion?)",
        fontsize=14, fontweight="bold", pad=20
    )
    plt.axis("off")
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "DAG_cyberball_moderated.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"DAG saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    draw_dag("forward")
    draw_dag("reverse")
    draw_cyberball_dag()