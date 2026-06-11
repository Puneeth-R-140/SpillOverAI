import json
import os
import random

def generate_ecosystem(output_path: str):
    """
    Generates a large-scale regional entrepreneurship ecosystem dataset with:
      - 15 Universities
      - 15 Incubators/Accelerators
      - 250 Startups across three distinct profiles (IP-Rich, VC-Heavies, Average Market Players)
      
    Funding distributions are heavily overlapped to ensure raw capital is a weak predictor
    on its own, highlighting that the engineered KSI network metric holds superior importance.
    """
    random.seed(42)
    
    nodes = []
    links = []
    
    # 1. Generate 15 Universities
    universities = [
        "IISc", "IITB", "IITD", "IITM", "RVCE", "PESU", "BMSCE", "MSRIT",
        "NITK", "BITS", "IIITB", "VIT", "MIT", "SJTU", "Stanford_India"
    ]
    for idx, name in enumerate(universities):
        tier = "Tier-1" if idx < 5 else ("Tier-2" if idx < 10 else "Tier-3")
        nodes.append({
            "id": name,
            "name": f"{name} Research Inst",
            "type": "University",
            "tier": tier
        })
        
    # 2. Generate 15 Incubators
    incubators = [
        "NSRCEL", "CIIE", "SINE", "T-Hub", "DERBI", "AIC-Jyothy", "Deshpande",
        "Gusec", "Startup-O", "Innovate-Kar", "Venture-Center", "Zone-Startups",
        "10000-Startups", "Kerala-Startup-Mission", "Coimbatore-TBI"
    ]
    for idx, name in enumerate(incubators):
        nodes.append({
            "id": name,
            "name": f"{name} Center",
            "type": "Incubator"
        })
        
    # 3. Generate 250 Startups
    # Distribution: 20% IP-Rich, 20% VC-Heavies, 60% Average Market Players
    for i in range(1, 251):
        startup_id = f"Startup_{i}"
        rand_val = random.random()
        
        if rand_val < 0.20:
            # IP-Rich Bootstrappers
            profile = "IP-Rich Bootstrapper"
            # High overlap range
            funding = round(random.uniform(30.0, 150.0), 2)
            patents = random.randint(3, 8)
            label = "High"
        elif rand_val < 0.40:
            # VC-Heavies
            profile = "VC-Heavy"
            # Overlapped funding
            funding = round(random.uniform(80.0, 220.0), 2)
            patents = random.randint(0, 1)
            label = "Low"
        else:
            # Average Market Players
            profile = "Average Market Player"
            # Overlapped funding
            funding = round(random.uniform(50.0, 180.0), 2)
            patents = random.randint(1, 2)
            label = "Medium"
            
        nodes.append({
            "id": startup_id,
            "name": f"Enterprise {i} ({profile})",
            "type": "Startup",
            "funding": funding,
            "patent_count": patents,
            "label": label,
            "profile": profile
        })
        
        # Connect Startups to institutions with different probabilities
        if profile == "IP-Rich Bootstrapper":
            # 80% probability to connect to top-tier university nodes
            for uni in universities:
                if random.random() < 0.25:
                    links.append({
                        "source": uni,
                        "target": startup_id,
                        "type": "TECH_TRANSFER" if random.random() < 0.5 else "JOINT_PATENT",
                        "weight": round(random.uniform(0.75, 0.99), 2)
                    })
            for inc in incubators:
                if random.random() < 0.08:
                    links.append({
                        "source": inc,
                        "target": startup_id,
                        "type": "INCUBATED_AT",
                        "weight": round(random.uniform(0.7, 0.95), 2)
                    })
                    
        elif profile == "VC-Heavy":
            # Less than 10% probability of having connections (isolated)
            for uni in universities:
                if random.random() < 0.003:
                    links.append({
                        "source": uni,
                        "target": startup_id,
                        "type": "TECH_TRANSFER",
                        "weight": round(random.uniform(0.1, 0.2), 2)
                    })
            for inc in incubators:
                if random.random() < 0.003:
                    links.append({
                        "source": inc,
                        "target": startup_id,
                        "type": "INCUBATED_AT",
                        "weight": round(random.uniform(0.1, 0.2), 2)
                    })
                    
        else:
            # Average Market Players - average probability
            for uni in universities:
                if random.random() < 0.03:
                    links.append({
                        "source": uni,
                        "target": startup_id,
                        "type": "TECH_TRANSFER",
                        "weight": round(random.uniform(0.3, 0.6), 2)
                    })
            for inc in incubators:
                if random.random() < 0.05:
                    links.append({
                        "source": inc,
                        "target": startup_id,
                        "type": "INCUBATED_AT",
                        "weight": round(random.uniform(0.4, 0.75), 2)
                    })
                    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({"nodes": nodes, "links": links}, f, indent=2)
        
    print(f"Dataset generated at '{output_path}' with {len(nodes)} nodes and {len(links)} links.")
