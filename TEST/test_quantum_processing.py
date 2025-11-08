import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sg_quantum_neuromorphic.core.quantum_processing_unit import QuantumProcessingUnit

class TestQuantumProcessingUnit:
    """Test cases for Quantum Processing Unit"""
    
    def test_initialization(self):
        """Test QPU initialization"""
        qpu = QuantumProcessingUnit(num_qubits=4)
        assert qpu.num_qubits == 4
        assert qpu.backend is not None
    
    def test_superposition_analysis(self):
        """Test quantum superposition analysis"""
        qpu = QuantumProcessingUnit(num_qubits=2)
        scenarios = [
            {'type': 'earthquake', 'magnitude': 7.5},
            {'type': 'flood', 'severity': 'high'}
        ]
        
        result = qpu.superposition_analysis(scenarios)
        assert 'quantum_probabilities' in result
        assert 'optimal_scenario' in result
        assert result['parallel_states_processed'] == 2
    
    def test_entanglement_coordination(self):
        """Test quantum entanglement coordination"""
        qpu = QuantumProcessingUnit()
        nodes = ['node1', 'node2', 'node3']
        
        result = qpu.entangle_nodes(nodes)
        assert result['entangled_nodes'] == 3
        assert 'quantum_state' in result

if __name__ == '__main__':
    pytest.main([__file__])
