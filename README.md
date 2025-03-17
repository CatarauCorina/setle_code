# SETLE Repository

This project provides implementations related to hierarchical task representation and outcome exploration using graph-based learning techniques as described in the paper "A representational framework for learning and encoding
structurally enriched trajectories in complex agent environments".


## Project Structure
The repository contains key components for task representation and neo4j database population.

### **1. Heterogeneous Graph GT Encoder**
- Located in the **`hetgraph_gt_encoder`** directory.
- Contains the **SET Encoder**, which is responsible for encoding hierarchical structures in the graph-based task representation.

### **2. Outcome Exploration & Database Population**
- Located in the **`outcome_exploration.py`** file.
- Includes scripts for populating the database and exploring outcomes using structured representations.
- You can view a demo of a few SET structures also by loadind outcomesmall.dump into an Aura DB or local Neo4j instance
- To get the SET representation of an episode in neo4j you can run this query:
  
```
 match (e:Episode)-[r:has_state]->(s:StateT)-[r1:has_object]->(o:ObjectConcept)
                     where elementId(e)="{you_set_id}"
                     with e,s, r1, o,r
                     optional match (o2:ObjectConcept)-[r2:contribute]->(a:ActionRepr) 
                     optional match (s)-[r3:influences]->(aff:Affordance)
                     match (a)-[r4:produces]->(aff) 
                     return e,s,r,r1,o,r2,a,r3,aff,r4 ,o2
```
## Getting Started
### **1. Clone the Repository**
```sh
 git clone https://github.com/CatarauCorina/setle_code.git
 cd setle_code
```

### **2. Install Dependencies**
Ensure you have Python and necessary libraries installed:
```sh
pip install -r requirements.txt
```

### **3. Running the SET Encoder**
To run the SET Encoder for hierarchical graph representation:
```sh
python hetgraph_gt_encoder/main.py
```

### **4. Running Database Population**
To populate the database and explore outcomes, you have to setup a neo4j db and run the scripts.


## Contact
For questions or collaboration, please open an issue in the repository.

---
**Author:** [Catarau Corina](https://github.com/CatarauCorina)
