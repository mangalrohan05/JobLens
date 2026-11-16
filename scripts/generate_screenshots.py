"""Generate visual UI screenshots and preview cards for the JobLens documentation."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def generate_search_screenshot():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    # Title & Header
    ax.text(0.04, 0.93, "JobLens — Relevance-Ranked Search", fontsize=15, fontweight="bold", color="#f8fafc")
    ax.text(0.04, 0.88, "Multi-field BM25F ranking with term contribution breakdown & filters", fontsize=9.5, color="#94a3b8")

    # Search bar box
    sb = patches.FancyBboxPatch((0.04, 0.77), 0.72, 0.07, boxstyle="round,pad=0.01", facecolor="#1e293b", edgecolor="#3b82f6", linewidth=1.5)
    ax.add_patch(sb)
    ax.text(0.06, 0.805, "machine learning engineer python pytorch", fontsize=10.5, color="#f8fafc", va="center")

    btn = patches.FancyBboxPatch((0.78, 0.77), 0.18, 0.07, boxstyle="round,pad=0.01", facecolor="#2563eb", edgecolor="none")
    ax.add_patch(btn)
    ax.text(0.87, 0.805, "BM25F Search", fontsize=10, fontweight="bold", color="#ffffff", ha="center", va="center")

    # Filter chips
    filters = ["Role: Data & AI", "Country: US", "Mode: Remote / Hybrid", "Exp: <= 5 yrs"]
    fx = 0.04
    for f in filters:
        chip = patches.FancyBboxPatch((fx, 0.70), 0.18, 0.04, boxstyle="round,pad=0.005", facecolor="#334155", edgecolor="none")
        ax.add_patch(chip)
        ax.text(fx + 0.09, 0.72, f, fontsize=8, color="#cbd5e1", ha="center", va="center")
        fx += 0.20

    # Job card 1
    c1 = patches.FancyBboxPatch((0.04, 0.38), 0.92, 0.28, boxstyle="round,pad=0.01", facecolor="#1e293b", edgecolor="#334155", linewidth=1)
    ax.add_patch(c1)
    border1 = patches.Rectangle((0.04, 0.38), 0.008, 0.28, facecolor="#3b82f6", edgecolor="none")
    ax.add_patch(border1)

    ax.text(0.065, 0.615, "#1. Machine Learning Engineer - Core AI Platform", fontsize=11.5, fontweight="bold", color="#f8fafc")
    ax.text(0.94, 0.615, "Score: 11.95", fontsize=9.5, fontweight="bold", color="#93c5fd", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e3a8a", edgecolor="none"))
    ax.text(0.065, 0.565, "Acme AI Labs  |  San Francisco, CA, US  |  3 - 5 yrs experience  |  $160,000 - $195,000", fontsize=8.5, color="#94a3b8")
    ax.text(0.065, 0.515, "Role Family: Data & AI  |  Mode: Hybrid  |  Field boosts: title=3.0, skills=2.0, desc=1.0", fontsize=8, color="#a5b4fc")

    # Term contribution table preview
    ax.text(0.065, 0.455, "BM25 Term Contribution:", fontsize=8, fontweight="bold", color="#cbd5e1")
    terms_txt = "machine: +3.41  |  learning: +3.28  |  engineer: +2.15  |  python: +1.89  |  pytorch: +1.22"
    ax.text(0.065, 0.415, terms_txt, fontsize=8, color="#38bdf8", family="monospace")

    # Job card 2
    c2 = patches.FancyBboxPatch((0.04, 0.07), 0.92, 0.27, boxstyle="round,pad=0.01", facecolor="#1e293b", edgecolor="#334155", linewidth=1)
    ax.add_patch(c2)
    border2 = patches.Rectangle((0.04, 0.07), 0.008, 0.27, facecolor="#3b82f6", edgecolor="none")
    ax.add_patch(border2)

    ax.text(0.065, 0.295, "#2. Senior ML & Deep Learning Engineer", fontsize=11.5, fontweight="bold", color="#f8fafc")
    ax.text(0.94, 0.295, "Score: 11.32", fontsize=9.5, fontweight="bold", color="#93c5fd", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e3a8a", edgecolor="none"))
    ax.text(0.065, 0.245, "Cortex Dynamics  |  Austin, TX, US  |  4+ yrs experience  |  $150,000 - $180,000", fontsize=8.5, color="#94a3b8")
    ax.text(0.065, 0.195, "Skills: python, pytorch, tensorflow, docker, kubernetes, mlops", fontsize=8, color="#6ee7b7")
    ax.text(0.065, 0.135, "Terms: learning (+3.28), machine (+3.12), python (+1.89), engineer (+2.15), pytorch (+0.88)", fontsize=8, color="#38bdf8", family="monospace")

    plt.tight_layout()
    out = DOCS_DIR / "screenshot_search.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out}")


def generate_recommendations_screenshot():
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    ax.text(0.04, 0.93, "JobLens — Personalized Recommendations & Skill Gap", fontsize=15, fontweight="bold", color="#f8fafc")
    ax.text(0.04, 0.88, "Candidate profile alignment, fit explanations, and top-N skill acquisition advice", fontsize=9.5, color="#94a3b8")

    # Profile summary bar
    pb = patches.FancyBboxPatch((0.04, 0.77), 0.92, 0.08, boxstyle="round,pad=0.01", facecolor="#1e293b", edgecolor="#10b981", linewidth=1.2)
    ax.add_patch(pb)
    ax.text(0.06, 0.81, "Profile: 3.5 yrs exp  |  Skills: python, aws, docker, sql, git  |  Preferred: Cloud / DevOps / Backend", fontsize=8.5, color="#f8fafc", va="center")

    # Aggregate Skill Gap cards
    ax.text(0.04, 0.72, "Top Skill Gaps in Recommended Postings (Top 10 Roles):", fontsize=9.5, fontweight="bold", color="#f8fafc")
    gaps = [("KUBERNETES", "70%"), ("TERRAFORM", "60%"), ("CI/CD", "50%"), ("LINUX", "40%"), ("FASTAPI", "30%")]
    gx = 0.04
    for skill, pct in gaps:
        gc = patches.FancyBboxPatch((gx, 0.58), 0.17, 0.11, boxstyle="round,pad=0.008", facecolor="#1e293b", edgecolor="#d97706", linewidth=1)
        ax.add_patch(gc)
        ax.text(gx + 0.085, 0.655, skill, fontsize=8, fontweight="bold", color="#fde68a", ha="center")
        ax.text(gx + 0.085, 0.61, pct, fontsize=12, fontweight="bold", color="#f59e0b", ha="center")
        gx += 0.19

    # Recommendation Card 1
    rc1 = patches.FancyBboxPatch((0.04, 0.12), 0.92, 0.42, boxstyle="round,pad=0.01", facecolor="#1e293b", edgecolor="#334155", linewidth=1)
    ax.add_patch(rc1)
    border = patches.Rectangle((0.04, 0.12), 0.008, 0.42, facecolor="#10b981", edgecolor="none")
    ax.add_patch(border)

    ax.text(0.065, 0.495, "#1. Senior Cloud & DevOps Infrastructure Engineer", fontsize=11.5, fontweight="bold", color="#f8fafc")
    ax.text(0.94, 0.495, "Match Score: 88.5%", fontsize=9.5, fontweight="bold", color="#6ee7b7", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#064e3b", edgecolor="none"))
    ax.text(0.065, 0.455, "CloudScale Inc.  |  Remote (US)  |  Requires: 3 - 6 yrs  |  $145,000 - $175,000", fontsize=8.5, color="#94a3b8")

    # Fit metrics row
    ax.text(0.065, 0.405, "Skill Coverage: 60%  |  BM25 Fit: 94%  |  Experience Fit: 100%  |  Location Fit: 100%", fontsize=8.5, fontweight="bold", color="#93c5fd")

    # Rationale
    ax.text(0.065, 0.355, "Rationale: Senior Cloud & DevOps at CloudScale Inc matches 3/5 required skills. Meets 3-6 yrs experience requirement. Matches Remote preference.", fontsize=8, style="italic", color="#cbd5e1")

    # Matched vs Missing skills
    ax.text(0.065, 0.285, "Matched Skills:", fontsize=8, fontweight="bold", color="#6ee7b7")
    ax.text(0.19, 0.285, "[✓ python]  [✓ aws]  [✓ docker]", fontsize=8, color="#a7f3d0", family="monospace")

    ax.text(0.065, 0.225, "Missing Skills (Gap):", fontsize=8, fontweight="bold", color="#f59e0b")
    ax.text(0.24, 0.225, "[✗ kubernetes]  [✗ terraform]", fontsize=8, color="#fde68a", family="monospace")

    ax.text(0.065, 0.155, "Actionable Tip: Adding Kubernetes and Terraform to your repertoire covers 100% of this role's stack.", fontsize=8, color="#38bdf8")

    plt.tight_layout()
    out = DOCS_DIR / "screenshot_recommendations.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out}")


def generate_market_insights_screenshot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    # Plot 1: Top in-demand skills
    ax1.set_facecolor("#1e293b")
    skills = ["Python", "SQL", "AWS", "Docker", "Java", "Kubernetes", "React", "Linux"]
    counts = [1420, 1180, 890, 740, 680, 610, 580, 520]
    bars1 = ax1.barh(skills[::-1], counts[::-1], color="#38bdf8", alpha=0.9, height=0.6)
    ax1.set_title("Most In-Demand Technical Skills", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax1.tick_params(colors="#cbd5e1", labelsize=8)
    ax1.grid(axis="x", linestyle="--", alpha=0.3, color="#64748b")
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # Plot 2: Median Salary by Role Family
    ax2.set_facecolor("#1e293b")
    families = ["Data & AI", "Cybersecurity", "Cloud & DevOps", "Software Eng.", "Product & Mgmt.", "Operations"]
    salaries = [155, 148, 142, 138, 130, 85]
    bars2 = ax2.bar(families, salaries, color="#10b981", alpha=0.9, width=0.55)
    ax2.set_title("Median Disclosed Salary by Role Family ($k)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("USD (Thousands)", color="#94a3b8", fontsize=8)
    ax2.tick_params(colors="#cbd5e1", labelsize=8)
    ax2.set_xticklabels(families, rotation=35, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    fig.suptitle("JobLens — Job Market Intelligence & Analytics Dashboard", color="#f8fafc", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out = DOCS_DIR / "screenshot_market_insights.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    generate_search_screenshot()
    generate_recommendations_screenshot()
    generate_market_insights_screenshot()
