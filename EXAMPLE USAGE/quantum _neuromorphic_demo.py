"""
Quantum Neuromorphic System Fabric Demo

SAFEWAY GUARDIAN | Nicolas E. Santiago | Nov 8, 2025
MIT LICENSE
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sg_quantum_neuromorphic import QuantumNeuromorphicFabric

def main():
    """Demonstrate Quantum Neuromorphic Fabric capabilities"""
    
    print("🚀 Initializing Quantum Neuromorphic System Fabric...")
    fabric = QuantumNeuromorphicFabric()
    
    # Create multiple disaster scenarios for parallel processing
    disaster_scenarios = [
        {
            'type': 'compound_disaster',
            'components': ['earthquake', 'tsunami', 'infrastructure_failure'],
            'population_affected': 1000000,
            'time_critical': True
        },
        {
            'type': 'cyber_physical_attack', 
            'components': ['power_grid', 'communication', 'transportation'],
            'sophistication': 'advanced',
            'coordinated': True
        },
        {
            'type': 'pandemic_outbreak',
            'components': ['biological', 'healthcare_strain', 'supply_chain'],
            'transmission_rate': 'high',
            'mutation_risk': True
        }
    ]
    
    print("🔮 Processing scenarios in quantum parallel...")
    results = fabric.parallel_reality_processing(disaster_scenarios)
    
    print("📊 Results:")
    print(f"- Scenarios processed: {results.get('parallel_states_processed', 'N/A')}")
    print(f"- Quantum advantage: {results.get('quantum_advantage', 'N/A')}")
    
    if 'collective_decision' in results:
        cd = results['collective_decision']
        print(f"- Collective intelligence: {cd.get('collective_confidence', 'N/A')}")
        print(f"- Emergent properties: {cd.get('emergent_properties', [])}")
    
    # Demonstrate neuromorphic learning
    print("\n🎓 Demonstrating neuromorphic learning...")
    training_data = [{'experience': f'scenario_{i}', 'outcome': 'success'} for i in range(10)]
    learning_result = fabric.neuromorphic_learning(training_data)
    
    print(f"Learning cycles: {learning_result.get('learning_cycles', 'N/A')}")
    print(f"Synaptic strengthening: {learning_result.get('synaptic_strengthening', 'N/A')}")
    
    # Demonstrate self-organization
    print("\n🌀 Demonstrating fabric self-organization...")
    org_result = fabric.fabric_self_organization()
    
    print(f"Network efficiency improvement: {org_result.get('efficiency_improvement', 'N/A')}")
    print(f"Self-healing capability: {org_result.get('self_healing_capability', 'N/A')}")

if __name__ == "__main__":
    main()
