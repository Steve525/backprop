package Backpropagation;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

public abstract class AbstractUnit extends AbstractNode {

	double _errorTerm;
	double _net;
	ArrayList<Double> _inputs;
	ArrayList<Double> _weights;
	Map<INode, Double> _nodesToWeights;
	protected int _id;
	protected double _biasWeight;
	double _biasDeltaWeight;
	Map<INode, Double> _nodesToDeltaWeights;
	
	public AbstractUnit() {
		_errorTerm = 0;
		_biasWeight = 0;
		_net = 0;
		_inputs = new ArrayList<Double>();
		_weights = new ArrayList<Double>();
		_nodesToWeights = new LinkedHashMap<INode, Double>();
		_biasDeltaWeight = 0;
		_nodesToDeltaWeights = new LinkedHashMap<INode, Double>();
	}
	
	public void setBiasWeight(double biasWeight) { _biasWeight = biasWeight; }
	
	public double getNet() { return _net; }
	
	public void receiveInputs() {
		_inputs.clear();
		_weights.clear();
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		while (iterator.hasNext()) {
			INode node = (INode)iterator.next();
			double output = node.getOutput();
			double weight = _nodesToWeights.get(node);
			_inputs.add(output);
			_weights.add(weight);
		}		
	}
	
	private void calculateNet() {
		_net = _biasWeight;
		for (int i = 0; i < _weights.size(); i++) {
			_net += _weights.get(i) * _inputs.get(i);
		}
	}
	
	public void calculateOutput() {
		calculateNet();
		_output = 1 / (1 + Math.exp(-1 * _net));
	}
	
	public abstract void calculateErrorTerm();
	
	public void updateWeights(double learningRate, double momentum, int iterations) {
		_biasDeltaWeight = (learningRate * _errorTerm * 1) + (momentum * _biasDeltaWeight);
		_biasWeight += _biasDeltaWeight;
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		double delta_weight = 0;
		while (iterator.hasNext()) {
			INode node = (INode)iterator.next();
			double output = node.getOutput();
			double weight = _nodesToWeights.get(node);
			double prevDeltaWeight = _nodesToDeltaWeights.get(node);
			delta_weight = (learningRate * _errorTerm * output) + (momentum * prevDeltaWeight);
			weight += delta_weight;
			_nodesToWeights.put(node, weight);
			_nodesToDeltaWeights.put(node, delta_weight);
		}
	}
	
	public void assignErrors() {
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		while (iterator.hasNext()) {
			HiddenUnit hiddenUnit = (HiddenUnit)iterator.next();
			double weight = _nodesToWeights.get(hiddenUnit);
			hiddenUnit.receiveError(weight, _errorTerm);
		}
	}
	
	public void setWeightMap (INode node, double weight) {
		_nodesToWeights.put(node, weight);
	}
	
	public void setDeltaWeightMap (INode node, double initWeight) {
		_nodesToDeltaWeights.put(node, initWeight);
	}
	
	public void printWeights() {
		System.out.println("Weight of Unit#" + this.getID() + ": ");
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		System.out.print("	" + _biasWeight + ", ");
		while (iterator.hasNext()) {
			INode node = iterator.next();
			double weight = _nodesToWeights.get(node);
			System.out.print("	" + weight + ", ");
		}
		System.out.println("");
		System.out.println("");
	}
	
	public String toStringWeights() {
		String weights = "";
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		while (iterator.hasNext()) {
			INode node = iterator.next();
			double weight = _nodesToWeights.get(node);
			weights += Double.toString(weight);
			weights += ",";
		}
		weights += "\n";
		return weights;
	}
	
	public void printErrorTerm() {
		System.out.print("	" + _errorTerm + ", ");
	}
	
	public int getID() { return _id; }

}
