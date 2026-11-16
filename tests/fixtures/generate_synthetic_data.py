"""SYNTHETIC TEST FIXTURE GENERATOR ONLY.

WARNING: This module generates purely synthetic, artificial test fixtures
strictly for unit testing offline components. It MUST NOT be used for real dataset
benchmarks, training, or evaluations.
"""

import random
from pathlib import Path
import pandas as pd

random.seed(42)

COMPANIES = [
    "Google", "Microsoft", "Amazon Web Services", "Meta Platforms", "Apple", "Netflix",
    "Stripe", "Spotify", "Databricks", "Snowflake", "OpenAI", "Airbnb", "Uber",
    "Salesforce", "Shopify", "Cloudflare", "GitHub", "Atlassian", "Twilio", "Palantir",
    "DeepMind", "NVIDIA", "Intel", "Cisco Systems", "Oracle", "IBM", "Adobe",
    "FinTech Labs", "HealthTech Innovations", "Autonomous Robotics Corp", "CyberShield Security",
    "NextGen Cloud Solutions", "Scale AI", "Anthropic", "Elastic", "MongoDB Inc.",
    "Cambridge BioTech", "Berlin AI Labs", "Toronto Quantum Labs", "Bengaluru Tech Hub",
]

LOCATIONS = [
    ("San Francisco, CA, USA", "US"),
    ("New York, NY, United States", "US"),
    ("Austin, TX, US", "US"),
    ("Seattle, WA", "US"),
    ("Boston, MA, United States", "US"),
    ("London, UK", "GB"),
    ("Cambridge, England, Great Britain", "GB"),
    ("Toronto, ON, Canada", "CA"),
    ("Vancouver, BC, Canada", "CA"),
    ("Berlin, Germany", "DE"),
    ("Munich, Germany", "DE"),
    ("Bengaluru, Karnataka, India", "IN"),
    ("Hyderabad, Telangana, India", "IN"),
    ("Remote - Worldwide", "Remote"),
    ("Remote, United States", "US"),
    ("Remote (Hybrid / London)", "GB"),
]

ROLE_TEMPLATES = [
    {
        "title": "Senior Software Engineer - Backend",
        "messy_titles": ["Sr. SWE [Remote]", "Senior Backend Dev", "Sr. Software Eng. (Backend)"],
        "skills": ["python", "go", "postgresql", "docker", "kubernetes", "microservices", "rest api", "aws", "git", "ci/cd", "system design", "problem solving"],
        "exp": "5-8 years",
        "salary_range": (140000, 195000),
    },
    {
        "title": "Machine Learning Engineer",
        "messy_titles": ["ML Engineer (m/w/d)", "Machine Learning Developer", "Lead ML / AI Engineer"],
        "skills": ["python", "pytorch", "tensorflow", "scikit-learn", "machine learning", "deep learning", "docker", "mlops", "numpy", "pandas", "airflow", "sql"],
        "exp": "3+ years",
        "salary_range": (130000, 185000),
    },
    {
        "title": "Full Stack Developer",
        "messy_titles": ["Fullstack Dev (100% Remote)", "Full Stack Software Engineer", "Sr. Fullstack Engineer"],
        "skills": ["react", "typescript", "node.js", "javascript", "html", "css", "postgresql", "tailwind css", "next.js", "graphql", "git", "teamwork"],
        "exp": "3-5 years",
        "salary_range": (110000, 160000),
    },
    {
        "title": "Data Scientist",
        "messy_titles": ["Data Scientist (Analytics)", "Senior Data Scientist", "Data Scientist [Hybrid]"],
        "skills": ["python", "r", "sql", "machine learning", "statistical analysis", "pandas", "data visualization", "tableau", "communication", "critical thinking"],
        "exp": "4+ years",
        "salary_range": (120000, 170000),
    },
    {
        "title": "DevOps & Cloud Engineer",
        "messy_titles": ["DevOps Engineer (m/w/d)", "Cloud Infrastructure Engineer", "Sr. DevOps / SRE"],
        "skills": ["aws", "terraform", "kubernetes", "docker", "linux", "bash", "ci/cd", "prometheus", "grafana", "ansible", "git", "site reliability engineering"],
        "exp": "4-7 years",
        "salary_range": (135000, 180000),
    },
    {
        "title": "Data Engineer",
        "messy_titles": ["Big Data Engineer", "Senior Data Engineer", "Data Pipeline Engineer"],
        "skills": ["apache spark", "python", "sql", "apache kafka", "dbt", "snowflake", "bigquery", "apache airflow", "etl", "data warehousing", "hadoop"],
        "exp": "3-6 years",
        "salary_range": (125000, 175000),
    },
    {
        "title": "Frontend Software Engineer",
        "messy_titles": ["Frontend Developer - React", "UI Engineer", "Lead Frontend Dev"],
        "skills": ["react", "javascript", "typescript", "redux", "html5", "css3", "tailwind css", "webpack", "responsive design", "jest", "cypress"],
        "exp": "2-5 years",
        "salary_range": (105000, 155000),
    },
    {
        "title": "Cybersecurity Analyst & Security Engineer",
        "messy_titles": ["Information Security Analyst", "Cybersecurity Engineer", "AppSec Specialist"],
        "skills": ["cybersecurity", "penetration testing", "network security", "firewalls", "siem", "tcp/ip", "cryptography", "oauth2", "compliance", "wireshark"],
        "exp": "3+ years",
        "salary_range": (115000, 165000),
    },
    {
        "title": "Junior Python Developer",
        "messy_titles": ["Jr. Python Dev (Full Time)", "Associate Software Developer", "Entry Level Software Engineer"],
        "skills": ["python", "django", "flask", "sql", "git", "rest api", "html", "problem solving", "continuous learning"],
        "exp": "0-2 years",
        "salary_range": (75000, 95000),
    },
    {
        "title": "NLP / AI Research Scientist",
        "messy_titles": ["Research Scientist - NLP", "Generative AI Specialist", "LLM Engineer"],
        "skills": ["natural language processing", "large language models", "generative ai", "pytorch", "transformers", "python", "deep learning", "langchain", "prompt engineering"],
        "exp": "5+ years",
        "salary_range": (160000, 240000),
    },
    {
        "title": "Mobile App Developer (iOS / Android)",
        "messy_titles": ["Mobile Developer", "iOS Engineer (Swift)", "Android Developer"],
        "skills": ["swift", "kotlin", "react native", "flutter", "ios", "android", "mobile app development", "git", "rest api", "teamwork"],
        "exp": "3-5 years",
        "salary_range": (115000, 160000),
    },
    {
        "title": "Product Manager - Technical",
        "messy_titles": ["Technical Product Manager", "Product Manager (Data Platforms)", "Senior PM"],
        "skills": ["project management", "agile", "scrum", "stakeholder management", "communication", "jira", "data visualization", "leadership", "strategic thinking"],
        "exp": "5-8 years",
        "salary_range": (130000, 185000),
    },
    {
        "title": "Software Development Engineer Intern",
        "messy_titles": ["Software Engineering Intern (Summer 2024)", "SWE Intern", "Backend Developer Trainee"],
        "skills": ["python", "java", "c++", "git", "data structures", "algorithms", "problem solving", "teamwork"],
        "exp": "no experience required",
        "salary_range": (50000, 70000),
    },
]

DESCRIPTION_TEMPLATES = [
    """<div>
<h3>About the Role</h3>
<p>We are seeking a talented <b>{role}</b> to join our innovative engineering team at <b>{company}</b> in {loc}. You will help design, build, and scale world-class software solutions.</p>

<h3>Key Responsibilities:</h3>
<ul>
  <li>Architect and maintain resilient, scalable applications and services.</li>
  <li>Collaborate with cross-functional teams including product managers and designers.</li>
  <li>Champion clean code practices, unit testing, and rigorous code reviews.</li>
  <li>Optimize system performance, latency, and operational reliability.</li>
</ul>

<h3>Qualifications & Experience:</h3>
<p>Requirements include <b>{exp}</b> in modern software engineering. Demonstrated hands-on expertise with <b>{skills_highlight}</b> is required.</p>
<p>Strong verbal communication, teamwork, and proactive analytical problem solving skills are highly valued.</p>
</div>""",

    """<p>At <b>{company}</b>, we are revolutionizing modern technology. We are currently looking for a passionate <b>{role}</b> based out of {loc}.</p>
<br>
<b>What You Will Do:</b>
<br>
- Build high-throughput pipelines and customer-facing interfaces.<br>
- Work closely with senior engineers to solve complex technical bottlenecks.<br>
- Drive automated CI/CD and deployment workflows.<br>
<br>
<b>Qualifications:</b><br>
- Requires {exp} of professional experience.<br>
- Core technical competencies: {skills_highlight}.<br>
- Familiarity with cloud technologies, containerization, and agile methodologies.<br>
- Strong attention to detail and high ownership.<br>
<br>
<i>{company} is an equal opportunity employer.</i>""",

    """<div>
<h2>Opportunity: {role}</h2>
<p>Company: {company} | Location: {loc}</p>
<div>
<h4>Position Overview:</h4>
<p>Join our mission-driven team. In this role, you will lead the implementation of mission-critical systems and collaborate across global hubs.</p>
<h4>Required Experience & Skills:</h4>
<ul>
  <li>Experience level: {exp} of relevant industry work.</li>
  <li>Must be proficient in {skills_highlight}.</li>
  <li>Solid grasp of system design, database optimization, and API design.</li>
  <li>Excellent critical thinking and leadership capabilities.</li>
</ul>
</div>
</div>"""
]


def generate_raw_postings(num_records: int = 1500) -> pd.DataFrame:
    """Generate diverse raw job postings DataFrame with realistic noise."""
    records = []

    for i in range(1, num_records + 1):
        template = random.choice(ROLE_TEMPLATES)
        company = random.choice(COMPANIES)
        loc_str, _ = random.choice(LOCATIONS)
        desc_tmpl = random.choice(DESCRIPTION_TEMPLATES)

        # 35% chance to use messy title
        if random.random() < 0.35 and template["messy_titles"]:
            title = random.choice(template["messy_titles"])
        else:
            title = template["title"]

        # Sample skills to highlight in text
        sampled_skills = random.sample(template["skills"], min(len(template["skills"]), random.randint(4, 8)))
        skills_highlight = ", ".join(s.upper() if len(s) <= 3 else s.title() for s in sampled_skills)

        # Experience string
        exp_str = template["exp"]

        # Generate HTML description
        desc_html = desc_tmpl.format(
            role=title,
            company=company,
            loc=loc_str,
            exp=exp_str,
            skills_highlight=skills_highlight,
        )

        # Dates over 2023-2024
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        year = random.choice([2023, 2024])
        posted_date = f"{year}-{month:02d}-{day:02d}"

        salary_min, salary_max = template["salary_range"]
        # Add random jitter
        jitter = random.randint(-5000, 15000)
        salary_min += jitter
        salary_max += jitter

        records.append({
            "job_id": f"job_{i:06d}",
            "job_title": title,
            "company_name": company,
            "job_location": loc_str,
            "job_description": desc_html,
            "posted_date": posted_date,
            "employment_type": "Full-time" if random.random() > 0.1 else "Contract",
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": "USD" if "US" in loc_str or "Remote" in loc_str else ("GBP" if "UK" in loc_str else "EUR"),
        })

    # Deliberately inject 40 duplicate records
    duplicates = [records[j].copy() for j in random.sample(range(len(records)), 40)]
    for dup in duplicates:
        # Give duplicate slightly different job_id or keep same
        dup["job_id"] = f"{dup['job_id']}_dup"
    records.extend(duplicates)

    # Deliberately inject 15 records with missing/null description or title
    for j in range(15):
        corrupt = records[j * 5].copy()
        corrupt["job_id"] = f"corrupt_{j:03d}"
        if j % 2 == 0:
            corrupt["job_description"] = None
        else:
            corrupt["job_title"] = ""
        records.append(corrupt)

    random.shuffle(records)
    return pd.DataFrame(records)


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "jobs_raw.csv"

    print(f"Generating realistic raw dataset at {csv_path}...")
    df = generate_raw_postings(num_records=1600)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Generated {len(df)} raw job postings in {csv_path}.")


if __name__ == "__main__":
    main()
