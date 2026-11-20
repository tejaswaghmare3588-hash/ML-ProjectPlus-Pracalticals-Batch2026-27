"""
HomeValue AI - Technical Diagram Generator
Generates high-resolution PNG diagrams for:
1. System Architecture
2. Project Workflow / Flowchart
3. UML Use Case Diagram
4. UML Class Diagram
5. UML Sequence Diagram
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGES_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(IMAGES_DIR, exist_ok=True)

# Styling constants
COLOR_NAVY = "#0B192C"
COLOR_BLUE = "#1E3E62"
COLOR_ACCENT = "#008DDA"
COLOR_GOLD = "#D4AF37"
COLOR_LIGHT_BG = "#F4F7FB"
COLOR_TEXT = "#1E293B"
COLOR_BOX_BG = "#FFFFFF"


def create_system_architecture():
    """Generates the System Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_facecolor(COLOR_LIGHT_BG)
    fig.patch.set_facecolor(COLOR_LIGHT_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    plt.title("HomeValue AI - System Architecture Diagram", fontsize=16, fontweight="bold", color=COLOR_NAVY, pad=15)

    # Layers
    layers = [
        ("Presentation Layer\n(Streamlit UI / Responsive Dashboard)", 0.5, 5.0, 11.0, 1.2, "#E0F2FE", COLOR_ACCENT),
        ("Application & Processing Layer\n(Feature Engineering, Validation & Routing)", 0.5, 3.2, 11.0, 1.2, "#FEF3C7", COLOR_GOLD),
        ("Machine Learning Engine\n(Sklearn Pipelines: GBDT, RF, DT, LR)", 0.5, 1.4, 5.2, 1.2, "#DCFCE7", "#16A34A"),
        ("Data & Persistence Tier\n(SQLite predictions.db & CSV Archives)", 6.3, 1.4, 5.2, 1.2, "#F3E8FF", "#9333EA")
    ]

    for title, x, y, w, h, bg, border in layers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_NAVY)

    # Connectors
    arrow_props = dict(facecolor=COLOR_NAVY, edgecolor=COLOR_NAVY, width=1.5, headwidth=7)
    ax.annotate("", xy=(6, 4.7), xytext=(6, 5.0), arrowprops=arrow_props)
    ax.annotate("", xy=(3.1, 2.85), xytext=(3.1, 3.2), arrowprops=arrow_props)
    ax.annotate("", xy=(8.9, 2.85), xytext=(8.9, 3.2), arrowprops=arrow_props)

    path = os.path.join(IMAGES_DIR, "system_architecture.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


def create_project_flowchart():
    """Generates the Machine Learning Pipeline Flowchart."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_facecolor(COLOR_LIGHT_BG)
    fig.patch.set_facecolor(COLOR_LIGHT_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    plt.title("HomeValue AI - Project ML & User Flowchart", fontsize=15, fontweight="bold", color=COLOR_NAVY, pad=15)

    steps = [
        ("Raw Multi-City Housing Data (32,963 records)", 5, 9.2, "#DBEAFE", "#2563EB"),
        ("Data Cleaning & Outlier Removal (31,103 clean records)", 5, 7.8, "#E0E7FF", "#4F46E5"),
        ("Feature Engineering: Locality, BHK, Baths, Age, Furnishing", 5, 6.4, "#FEF3C7", "#D97706"),
        ("Preprocessing Pipeline (StandardScaler + OneHotEncoder)", 5, 5.0, "#FEF08A", "#CA8A04"),
        ("Model Comparison: Linear, Decision Tree, RF, Gradient Boosting", 5, 3.6, "#DCFCE7", "#16A34A"),
        ("Best Model Selected & Joblib Pipeline Serialized", 5, 2.2, "#CCFBF1", "#0D9488"),
        ("Streamlit Web Application & SQLite Prediction Logging", 5, 0.8, "#F3E8FF", "#9333EA")
    ]

    for text, cx, cy, bg, border in steps:
        rect = patches.FancyBboxPatch((cx - 4.0, cy - 0.45), 8.0, 0.9, boxstyle="round,pad=0.15", facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=10, fontweight="bold", color=COLOR_NAVY)

    for i in range(len(steps) - 1):
        ax.annotate("", xy=(5, steps[i+1][2] + 0.5), xytext=(5, steps[i][2] - 0.5),
                    arrowprops=dict(facecolor=COLOR_NAVY, edgecolor=COLOR_NAVY, width=1.5, headwidth=6))

    path = os.path.join(IMAGES_DIR, "project_flowchart.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


def create_uml_usecase():
    """Generates the UML Use Case Diagram."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.set_facecolor(COLOR_LIGHT_BG)
    fig.patch.set_facecolor(COLOR_LIGHT_BG)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    plt.title("HomeValue AI - UML Use Case Diagram", fontsize=15, fontweight="bold", color=COLOR_NAVY, pad=15)

    # Actor box
    ax.text(1.2, 3.5, "User / Buyer /\nReal Estate Agent", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_NAVY,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E2E8F0", edgecolor=COLOR_NAVY, linewidth=2))

    # Boundary Box
    system_rect = patches.Rectangle((3.0, 0.5), 7.5, 6.0, fill=False, edgecolor=COLOR_ACCENT, linewidth=2.5, linestyle="--")
    ax.add_patch(system_rect)
    ax.text(6.75, 6.2, "HomeValue AI System Boundary", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_ACCENT)

    use_cases = [
        ("Input Property Specs\n(City, Locality, BHK, Area, etc.)", 6.75, 5.2),
        ("Generate House Price Estimate\n& Prediction Interval", 6.75, 4.1),
        ("Explore Interactive Visualizations\n(Distribution & Price vs Area)", 6.75, 3.0),
        ("Review Historical Estimates\n& Export CSV Records", 6.75, 1.9),
        ("Inspect Machine Learning\nModel Comparison Metrics", 6.75, 0.9)
    ]

    for label, cx, cy in use_cases:
        ellipse = patches.Ellipse((cx, cy), 5.5, 0.8, facecolor="#F8FAFC", edgecolor=COLOR_NAVY, linewidth=1.5)
        ax.add_patch(ellipse)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color=COLOR_NAVY)
        ax.plot([2.2, cx - 2.6], [3.5, cy], color=COLOR_NAVY, linewidth=1.2)

    path = os.path.join(IMAGES_DIR, "uml_usecase.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


def create_uml_class():
    """Generates the UML Class Diagram."""
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_facecolor(COLOR_LIGHT_BG)
    fig.patch.set_facecolor(COLOR_LIGHT_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    plt.title("HomeValue AI - UML Class Diagram", fontsize=15, fontweight="bold", color=COLOR_NAVY, pad=15)

    classes = [
        ("PropertyInput", 0.6, 3.8, 3.2, 3.2,
         "+ city: str\n+ locality: str\n+ area_sqft: float\n+ bhk: int\n+ bathrooms: int\n+ balconies: int\n+ property_age: int\n+ parking: str\n+ furnishing_status: str\n+ property_type: str",
         "+ to_dataframe(): DataFrame\n+ validate(): bool"),

        ("HousePricePipeline", 4.4, 3.8, 3.2, 3.2,
         "- preprocessor: ColumnTransformer\n- regressor: BaseEstimator\n- residual_std: float\n- metadata: dict",
         "+ load_pipeline(): Pipeline\n+ predict(input_df): float\n+ predict_interval(val): tuple\n+ get_metrics(): dict"),

        ("DatabaseHandler", 8.2, 3.8, 3.2, 3.2,
         "- db_path: str\n- connection: sqlite3.Conn",
         "+ init_db(): void\n+ save_prediction(...): int\n+ get_prediction_history(): DF\n+ get_summary_metrics(): dict\n+ export_predictions_csv(): str"),

        ("HomeValueApp (Streamlit UI)", 3.0, 0.4, 6.0, 2.4,
         "- current_page: str\n- model_pipeline: Pipeline\n- db_handler: DatabaseHandler",
         "+ render_home_page(): void\n+ render_prediction_form(): void\n+ render_analytics(): void\n+ render_history(): void\n+ render_model_performance(): void")
    ]

    for name, x, y, w, h, attrs, methods in classes:
        # Box header
        hdr = patches.Rectangle((x, y + h - 0.6), w, 0.6, facecolor=COLOR_NAVY, edgecolor=COLOR_NAVY)
        body = patches.Rectangle((x, y), w, h - 0.6, facecolor="#FFFFFF", edgecolor=COLOR_NAVY, linewidth=1.5)
        ax.add_patch(body)
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 0.3, name, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#FFFFFF")
        ax.text(x + 0.1, y + h - 0.8, attrs, ha="left", va="top", fontsize=8.5, color=COLOR_TEXT, family="monospace")
        ax.plot([x, x + w], [y + 1.2, y + 1.2], color=COLOR_NAVY, linewidth=1)
        ax.text(x + 0.1, y + 1.0, methods, ha="left", va="top", fontsize=8.5, color=COLOR_TEXT, family="monospace")

    path = os.path.join(IMAGES_DIR, "uml_class.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


def create_uml_sequence():
    """Generates the UML Sequence Diagram."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.set_facecolor(COLOR_LIGHT_BG)
    fig.patch.set_facecolor(COLOR_LIGHT_BG)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    plt.title("HomeValue AI - UML Sequence Diagram (Prediction Flow)", fontsize=15, fontweight="bold", color=COLOR_NAVY, pad=15)

    actors = ["User", "Streamlit UI", "ML Pipeline", "SQLite DB"]
    xs = [1.5, 4.0, 7.0, 9.5]

    for name, x in zip(actors, xs):
        rect = patches.FancyBboxPatch((x - 1.0, 6.0), 2.0, 0.6, boxstyle="round,pad=0.1", facecolor=COLOR_NAVY, edgecolor=COLOR_GOLD, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 6.3, name, ha="center", va="center", fontsize=10, fontweight="bold", color="#FFFFFF")
        ax.plot([x, x], [0.8, 6.0], color=COLOR_NAVY, linestyle="--", linewidth=1.2)

    messages = [
        (1.5, 4.0, 5.3, "1. Submit property parameters", True),
        (4.0, 7.0, 4.5, "2. Format schema & invoke predict()", True),
        (7.0, 7.0, 3.8, "3. Feature scaling & GBDT inference", False),
        (7.0, 4.0, 3.0, "4. Return price & confidence interval", True),
        (4.0, 9.5, 2.2, "5. save_prediction(...) log record", True),
        (4.0, 1.5, 1.4, "6. Display price card, range & metric badges", True)
    ]

    for x1, x2, y, msg, is_arrow in messages:
        if is_arrow:
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(facecolor=COLOR_NAVY, edgecolor=COLOR_NAVY, width=1.2, headwidth=5))
        ax.text((x1 + x2) / 2, y + 0.15, msg, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_NAVY)

    path = os.path.join(IMAGES_DIR, "uml_sequence.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {path}")


def generate_all_diagrams():
    """Generates all 5 technical diagrams."""
    print("Generating technical diagrams...")
    create_system_architecture()
    create_project_flowchart()
    create_uml_usecase()
    create_uml_class()
    create_uml_sequence()
    print("All diagrams generated successfully!")


if __name__ == "__main__":
    generate_all_diagrams()
