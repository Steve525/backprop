package Backpropagation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public abstract class AbstractUnit extends AbstractNode {

	double _errorTerm;
	double _net;
	ArrayList<Double> _inputs;
	ArrayList<Double> _weights;
	Map<INode, Double> _nodesToWeights;
	protected int _number;
	protected double _biasWeight;
	
	public AbstractUnit() {
		_errorTerm = 0;
		_biasWeight = 0;
		_net = 0;
		_inputs = new ArrayList<Double>();
		_weights = new ArrayList<Double>();
		_nodesToWeights = new HashMap<INode, Double>();
	}
	
	public void setBiasWeight(double biasWeight) { _biasWeight = biasWeight; }
	
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
		_net = _biasWeight * 1;
		for (int i = 0; i < _weights.size(); i++) {
			_net += _weights.get(i) * _inputs.get(i);
		}
	}
	
	public void calculateOutput() {
		calculateNet();
		_output = 1 / (1 + Math.exp(-1 * _net));
	}
	
	public abstract void calculateErrorTerm();
	
	public void updateWeights(double learningRate) {
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		while (iterator.hasNext()) {
			INode node = (INode)iterator.next();
			double output = node.getOutput();
			double weight = _nodesToWeights.get(node);
			double delta_weight = learningRate * _errorTerm * output;
			weight += delta_weight;
			_nodesToWeights.put(node, weight);
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
	
	public void printWeights() {
		System.out.println("Weight of Unit#" + this.getNumber() + ": ");
		Iterator<INode> iterator = _nodesToWeights.keySet().iterator();
		while (iterator.hasNext()) {
			INode node = iterator.next();
			double weight = _nodesToWeights.get(node);
			System.out.print(weight + ", ");
		}
		System.out.println("");
	}
	
	public int getNumber() { return _number; }

}
