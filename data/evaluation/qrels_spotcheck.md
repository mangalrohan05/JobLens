# JobLens Qrels Spot-Check Form

This document contains the top-5 retrieved postings for 5 diverse evaluation queries (3 technical, 2 non-technical) across two retrieval systems:
1. **Keyword Baseline Ranker** (raw term-frequency count in OR mode)
2. **BM25F Default Ranker** (field-weighted Robertson-Spärck Jones BM25)

**Instructions for Project Owner / Evaluator**:
- Review each retrieved posting title, company, and location.
- Record your independent manual assessment in the **Manual Verdict** column (`[X] Rel` or `[X] Irrel`).
- Compare your verdict with the **Rule Grade** (2 = Highly Relevant, 1 = Partially Relevant, 0 = Irrelevant) to evaluate how well rule-based labeling reflects human judgment.

## Query [q01]: "machine learning engineer python pytorch"
**Target Role Family**: Data & AI  
**Core Skills**: python, pytorch, machine learning, tensorflow  

### System 1: BM25F (Default Multi-Field)
| Rank | Doc ID | Job Title | Company | Location | BM25 Score | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3757454354` | Machine Learning Engineer - $175k+ | nan | New York City Metropolitan Area | 11.95 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3757449955` | Software Engineer - Quantitative | Aplitrak | Boston, MA | 11.77 | **0** | [ ] Rel / [ ] Irrel |
| 3 | `3749343196` | W2 Only - Backend Software Engineer (python) | nan | United States | 11.32 | **0** | [ ] Rel / [ ] Irrel |
| 4 | `3755581424` | Mlops Generative AI Engineer | nan | Milpitas, CA | 11.32 | **1** | [ ] Rel / [ ] Irrel |
| 5 | `3749351488` | Cloud Engineer | nan | Dearborn, MI | 11.29 | **0** | [ ] Rel / [ ] Irrel |

### System 2: Keyword Baseline Ranker (Term Count)
| Rank | Doc ID | Job Title | Company | Location | Term Count | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3701198711` | Mentor - Machine Learning Career Track (part-time/contract) | nan | United States | 44 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3757454354` | Machine Learning Engineer - $175k+ | nan | New York City Metropolitan Area | 43 | **2** | [ ] Rel / [ ] Irrel |
| 3 | `3694102195` | Senior ML Specialist Solutions Architect, ML Service Solution Architects | Amazon | Santa Clara, CA | 33 | **0** | [ ] Rel / [ ] Irrel |
| 4 | `3697377900` | Machine Learning Scientist I | Cambiahealth | Portland, OR | 33 | **2** | [ ] Rel / [ ] Irrel |
| 5 | `3697379856` | Machine Learning Scientist I | Cambiahealth | Utah, United States | 33 | **2** | [ ] Rel / [ ] Irrel |

---

## Query [q04]: "cloud devops engineer terraform kubernetes aws"
**Target Role Family**: Cloud & DevOps  
**Core Skills**: terraform, kubernetes, aws, docker, ci/cd  

### System 1: BM25F (Default Multi-Field)
| Rank | Doc ID | Job Title | Company | Location | BM25 Score | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3749345480` | AWS DevOps Engineer ( Onsite, Mon to Fri) | nan | Mountain View, CA | 17.88 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3757700106` | Senior DevOps Engineer | nan | Charlotte, NC | 16.22 | **1** | [ ] Rel / [ ] Irrel |
| 3 | `3757435583` | Senior DevOps Engineer | nan | United States | 15.92 | **1** | [ ] Rel / [ ] Irrel |
| 4 | `3757491052` | Senior AWS DevOps Engineer | Earlywarning | San Francisco, CA | 15.24 | **2** | [ ] Rel / [ ] Irrel |
| 5 | `3757492028` | Senior AWS DevOps Engineer | Earlywarning | Scottsdale, AZ | 15.24 | **2** | [ ] Rel / [ ] Irrel |

### System 2: Keyword Baseline Ranker (Term Count)
| Rank | Doc ID | Job Title | Company | Location | Term Count | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3693057176` | Manufacturing Cloud Technical Architect | nan | United States | 33 | **0** | [ ] Rel / [ ] Irrel |
| 2 | `3755588635` | Cloud Cyber Security Practice Lead | Www2 | Arlington, VA | 33 | **0** | [ ] Rel / [ ] Irrel |
| 3 | `3699059394` | Azure Cloud Engineer (o365) | nan | Richmond, VA | 32 | **1** | [ ] Rel / [ ] Irrel |
| 4 | `3756131363` | Presales Cloud/devops Solutions Architect- Remote | Insight | Raleigh, NC | 32 | **0** | [ ] Rel / [ ] Irrel |
| 5 | `3757470425` | Amazon Web Services (AWS) Tools Subject Matter Expert | nan | United States | 32 | **0** | [ ] Rel / [ ] Irrel |

---

## Query [q08]: "full stack software developer react python"
**Target Role Family**: Software Engineering  
**Core Skills**: python, react, javascript, typescript, html  

### System 1: BM25F (Default Multi-Field)
| Rank | Doc ID | Job Title | Company | Location | BM25 Score | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3755587247` | Full Stack Software Engineer | nan | Arlington, VA | 11.96 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3693051735` | Full Stack Angular Developer (w2 Only) | nan | Beaverton, OR | 11.37 | **0** | [ ] Rel / [ ] Irrel |
| 3 | `3749347122` | Senior Full Stack Java Developer- React (10+ Yrs Exp) | nan | Pennsylvania, United States | 11.34 | **0** | [ ] Rel / [ ] Irrel |
| 4 | `3754761827` | Full Stack Software Engineer | nan | Santa Clara, CA | 10.91 | **2** | [ ] Rel / [ ] Irrel |
| 5 | `3755549845` | Application Developer | Atos | Phoenix, AZ | 10.75 | **1** | [ ] Rel / [ ] Irrel |

### System 2: Keyword Baseline Ranker (Term Count)
| Rank | Doc ID | Job Title | Company | Location | Term Count | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3697350743` | Application Software Developer Lll | App | Chicago, IL | 44 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3749346453` | Adaptive Optics Senior Software Engineer | nan | Pasadena, CA | 37 | **2** | [ ] Rel / [ ] Irrel |
| 3 | `3694151722` | Lead Full Stack Developer | nan | Brooklyn Center, MN | 30 | **2** | [ ] Rel / [ ] Irrel |
| 4 | `3757722602` | Avionics Software Engineer | Textron | Fort Worth, TX | 29 | **2** | [ ] Rel / [ ] Irrel |
| 5 | `3757920617` | Full Stack Developer - Api Integrations | nan | United States | 29 | **2** | [ ] Rel / [ ] Irrel |

---

## Query [q16]: "registered nurse icu healthcare patient care"
**Target Role Family**: Operations & Other  
**Core Skills**: critical care, nursing, patient care, healthcare  

### System 1: BM25F (Default Multi-Field)
| Rank | Doc ID | Job Title | Company | Location | BM25 Score | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3757483861` | Registered Nurse - Rn - Intensive Care Unit - Icu | nan | Loma Linda, CA | 13.28 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3757490258` | Registered Nurse - Rn - Intensive Care Unit - Icu | nan | Moreno Valley, CA | 13.27 | **2** | [ ] Rel / [ ] Irrel |
| 3 | `3694121425` | Travel Nurse Rn - Icu - Intensive Care Unit - $1,908 Per Week - 990409 | Click | Nashville, TN | 13.09 | **2** | [ ] Rel / [ ] Irrel |
| 4 | `3749372759` | Travel Nurse Rn - Icu - Intensive Care Unit - $2,320 Per Week - A0f3m000012wxdaeaq | Click | Columbus, OH | 13.08 | **2** | [ ] Rel / [ ] Irrel |
| 5 | `3757484792` | Registered Nurse - Icu - Hospital | nan | Torrance, CA | 12.89 | **2** | [ ] Rel / [ ] Irrel |

### System 2: Keyword Baseline Ranker (Term Count)
| Rank | Doc ID | Job Title | Company | Location | Term Count | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3757479468` | Registered Nurse (rn) - Long Term Acute Care - 1068790 | Click | Temple, TX | 215 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3699416125` | Rn - Medical Intermediate - Days/nights | Ssmhealth | Greater Madison Area | 126 | **2** | [ ] Rel / [ ] Irrel |
| 3 | `3699057241` | Ambulatory Care Nurse - **gyn/gyn Exp Highly Preferred** | Nyu | New York, NY | 113 | **1** | [ ] Rel / [ ] Irrel |
| 4 | `3757486906` | Registered Nurse, Rn - Psychiatry | Tuftsmedicine | Boston, MA | 108 | **2** | [ ] Rel / [ ] Irrel |
| 5 | `3699414647` | Clinical Nurse-msh- Full Time Nights- 9 West | Mountsinai | United States | 106 | **2** | [ ] Rel / [ ] Irrel |

---

## Query [q21]: "financial analyst financial modeling excel forecasting"
**Target Role Family**: Operations & Other  
**Core Skills**: financial modeling, excel, forecasting, variance analysis, finance  

### System 1: BM25F (Default Multi-Field)
| Rank | Doc ID | Job Title | Company | Location | BM25 Score | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3754503407` | Senior Financial Analyst | Capitalsearch | Vienna, VA | 9.12 | **2** | [ ] Rel / [ ] Irrel |
| 2 | `3756131156` | Senior Financial Analyst | nan | Provo, UT | 8.94 | **2** | [ ] Rel / [ ] Irrel |
| 3 | `3693047420` | Senior Financial Analyst | nan | Atlanta, GA | 8.91 | **2** | [ ] Rel / [ ] Irrel |
| 4 | `3693050670` | Financial Planning Analyst | nan | Newark, NJ | 8.87 | **0** | [ ] Rel / [ ] Irrel |
| 5 | `3757712169` | Investment Analyst (real Estate) | nan | New York City Metropolitan Area | 8.72 | **1** | [ ] Rel / [ ] Irrel |

### System 2: Keyword Baseline Ranker (Term Count)
| Rank | Doc ID | Job Title | Company | Location | Term Count | Rule Grade | Manual Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `3755591342` | Financial Director | Aplitrak | New York, NY | 47 | **0** | [ ] Rel / [ ] Irrel |
| 2 | `3755586060` | Chief Financial Officer (cfo) Mater Dei High School | nan | Santa Ana, CA | 39 | **0** | [ ] Rel / [ ] Irrel |
| 3 | `3757734067` | Senior Director, Finance | Phh | Washington, DC | 35 | **0** | [ ] Rel / [ ] Irrel |
| 4 | `3693047461` | Financial Controller | nan | United States | 34 | **0** | [ ] Rel / [ ] Irrel |
| 5 | `3693050670` | Financial Planning Analyst | nan | Newark, NJ | 33 | **0** | [ ] Rel / [ ] Irrel |

---
