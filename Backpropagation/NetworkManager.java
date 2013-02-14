package Backpropagation;

import java.util.ArrayList;

public class NetworkManager {
	
	private ArrayList<INode> _inputLayer;
	private ArrayList<ArrayList<INode>> _hiddenLayers;
	private ArrayList<INode> _outputLayer;
	private double _learningRate;
	private double _momentum;
	private double _mse;
	private int _unitCount;

	public NetworkManager(double learningRate, double momentum) {
		_unitCount = 0;
		_learningRate = learningRate;
		_momentum = momentum;
		_mse = 0;
		_inputLayer = new ArrayList<INode>();
		_hiddenLayers = new ArrayList<ArrayList<INode>>();
		_outputLayer = new ArrayList<INode>();
	}
	
	public void initializeNetwork(int n_in, int n_hidden, int n_hiddenLayers, int n_output) {
		// Initialize input layer
		for (int i = 0; i < n_in; i++) {
			INode inputNode = new InputNode();
			_inputLayer.add(inputNode);
		}
		// Initialize hidden layers
		for (int i = 0; i < n_hiddenLayers; i++) {
			ArrayList<INode> hiddenLayer = new ArrayList<INode>();
			_hiddenLayers.add(hiddenLayer);
			for (int j = 0; j < n_hidden; j++) {
				_unitCount++;
				INode hiddenUnit = new HiddenUnit(_unitCount);
				_hiddenLayers.get(i).add(hiddenUnit);
			}
		}
		_unitCount++;
		// Initialize output layers
		for (int i = 0; i < n_output; i++) {
			INode outputUnit = new OutputUnit(_unitCount);
			_unitCount++;
			_outputLayer.add(outputUnit);
		}
		
		// Initialize weights of all layers to a random value between -0.05 and 0.05
		initializeWeights (0.05, -0.05);
	}
	
	public void initializeNetwork() {
		INode inputNode1 = new InputNode();
		INode inputNode2 = new InputNode();
		_inputLayer.add(inputNode1);
		_inputLayer.add(inputNode2);
		
		ArrayList<INode> hiddenLayer1 = new ArrayList<INode>();
		ArrayList<INode> hiddenLayer2 = new ArrayList<INode>();
		_hiddenLayers.add(hiddenLayer1);
		_hiddenLayers.add(hiddenLayer2);
		
		INode hiddenUnit1 = new HiddenUnit(1);
		((AbstractUnit) hiddenUnit1).setBiasWeight(0.1);
		((AbstractUnit) hiddenUnit1).setWeightMap(inputNode1, 0.2);
		((AbstractUnit) hiddenUnit1).setWeightMap(inputNode2, -0.1);
		INode hiddenUnit2 = new HiddenUnit(2);
		((AbstractUnit) hiddenUnit2).setBiasWeight(-0.2);
		((AbstractUnit) hiddenUnit2).setWeightMap(inputNode1, 0.3);
		((AbstractUnit) hiddenUnit2).setWeightMap(inputNode2, -0.3);
		_hiddenLayers.get(0).add(hiddenUnit1);
		_hiddenLayers.get(0).add(hiddenUnit2);
		INode hiddenUnit3 = new HiddenUnit(3);
		((AbstractUnit) hiddenUnit3).setWeightMap(, -0.2);
		((AbstractUnit) hiddenUnit3).setWeightMap(, 0.3);
		((AbstractUnit) hiddenUnit3).setWeightMap(, -0.3);
		INode hiddenUnit4 = new HiddenUnit(4);
		((AbstractUnit) hiddenUnit4).setWeightMap(, -0.2);
		((AbstractUnit) hiddenUnit4).setWeightMap(, 0.3);
		((AbstractUnit) hiddenUnit4).setWeightMap(, -0.3);
		_hiddenLayers.get(1).add(hiddenUnit3);
		_hiddenLayers.get(1).add(hiddenUnit4);
		
		INode outputUnit1 = new OutputUnit(5);
		INode outputUnit2 = new OutputUnit(6);
		_outputLayer.add(outputUnit1);
		_outputLayer.add(outputUnit2);
	}
	
	public double getMSE() {
		return 0;
	}
	
	public void propagateInputForward (double[] inputs, double[] target) {
		giveInputNodesInput(inputs);
		giveOutputUnitsTargets(target);
		for (int i = 0; i < _hiddenLayers.size(); i++) {
			ArrayList<INode> hiddenLayer = _hiddenLayers.get(i);
			for (int j = 0; j < hiddenLayer.size(); j++) {
				HiddenUnit hiddenUnit = (HiddenUnit) hiddenLayer.get(j);
				hiddenUnit.receiveInputs();
				hiddenUnit.calculateOutput();
			}
		}
		
		for (int i = 0; i < _outputLayer.size(); i++) {
			OutputUnit outputUnit = (OutputUnit) _outputLayer.get(i);
			outputUnit.receiveInputs();
			outputUnit.calculateOutput();
		}
	}
	
	private void giveInputNodesInput (double[] inputs) {
		for (int i = 0; i < _inputLayer.size(); i++) {
			InputNode inputNode = (InputNode) _inputLayer.get(i);
			inputNode.setInput(inputs[i]);
		}
	}
	
	private void giveOutputUnitsTargets (double[] target) {
		for (int i = 0; i < _outputLayer.size(); i++) {
			OutputUnit outputUnit = (OutputUnit) _outputLayer.get(i);
			outputUnit.setTarget(target[i]);
		}
	}
	
	public void propagateErrorsBack() {
		calculateOutputUnitErrors();
		calculateHiddenUnitErrors();
		updateAllWeights();
	}
	
	private void calculateOutputUnitErrors() {
		for (int i = 0; i < _outputLayer.size(); i++) {
			OutputUnit outputUnit = (OutputUnit) _outputLayer.get(i);
			outputUnit.calculateErrorTerm();
			outputUnit.assignErrors();
		}
	}
	
	private void calculateHiddenUnitErrors() {
		for (int i = 0; i < _hiddenLayers.size() - 1; i++) {
			ArrayList<INode> hiddenLayer = _hiddenLayers.get(_hiddenLayers.size() - (i + 1));
			for (int j = 0; j < hiddenLayer.size(); j++) {
				HiddenUnit hiddenUnit = (HiddenUnit) hiddenLayer.get(j);
				hiddenUnit.calculateErrorTerm();
				hiddenUnit.assignErrors();
			}
		}
		
		ArrayList<INode> firstHiddenLayer = _hiddenLayers.get(0);
		for (int i = 0; i <firstHiddenLayer.size(); i++) {
			HiddenUnit anotherHiddenUnit = (HiddenUnit) firstHiddenLayer.get(i);
			anotherHiddenUnit.calculateErrorTerm();
		}
	}
	
	private void updateAllWeights() {
		
	}
	
	private void initializeWeights (double Max, double Min) {
		initializeWeightsOutputLayer(Max, Min);
		initializeWeightsHiddenLayers (Max, Min);
		initializeWeightsHiddenInputLayer (Max, Min);
	}
	
	private void initializeWeightsOutputLayer (double Max, double Min) {
		ArrayList<INode> lastHiddenLayer = _hiddenLayers.get(_hiddenLayers.size() - 1);
		for (int i = 0; i < _outputLayer.size(); i++) {
			OutputUnit outputUnit = (OutputUnit) _outputLayer.get(i);
			for (int j = 0; j < lastHiddenLayer.size(); j++) {
				INode hiddenUnit = lastHiddenLayer.get(j);
				outputUnit.setWeightMap(hiddenUnit, getRandomWeight(Max, Min));
			}
		}
	}
	
	private void initializeWeightsHiddenLayers (double Max, double Min) {
		for (int k = 0; k < _hiddenLayers.size() - 1; k++) {
			ArrayList<INode> hiddenLayer = _hiddenLayers.get(_hiddenLayers.size() - (k + 1));
			ArrayList<INode> nextHiddenLayer = _hiddenLayers.get(_hiddenLayers.size() - (k + 2));
			for (int i = 0; i < hiddenLayer.size(); i++) {
				HiddenUnit hiddenUnit = (HiddenUnit) hiddenLayer.get(i); //hidden unit from hidden layer
				for (int j = 0; j < nextHiddenLayer.size(); j++) {
					INode nextHiddenUnit = nextHiddenLayer.get(j);
					hiddenUnit.setWeightMap(nextHiddenUnit, getRandomWeight(Max, Min));
				}
			}
		}
	}
	
	private void initializeWeightsHiddenInputLayer (double Max, double Min) {
		ArrayList<INode> firstHiddenLayer = _hiddenLayers.get(0);
		for (int i = 0; i < firstHiddenLayer.size(); i++) {
			HiddenUnit hiddenUnit = (HiddenUnit) firstHiddenLayer.get(i);
			for (int j = 0; j < _inputLayer.size(); j++) {
				INode inputNode = _inputLayer.get(j);
				hiddenUnit.setWeightMap(inputNode, getRandomWeight(Max, Min));
			}
		}
	}
	
	private double getRandomWeight (double Max, double Min) {
		double weight = Min + (Math.random() * (Max - Min));
		return weight;
	}
	
	public void printWeights () {
		System.out.println("HIDDEN LAYER: ");
		for (int i = 0; i < _hiddenLayers.size(); i++) {
			ArrayList<INode> hiddenLayer = _hiddenLayers.get(i);
			for (int j = 0; j < hiddenLayer.size(); j++) {
				((AbstractUnit)hiddenLayer.get(j)).printWeights();
			}
		}
		System.out.println("OUTPUT LAYER: ");
		for (int j = 0; j < _outputLayer.size(); j++) {
			((AbstractUnit)_outputLayer.get(j)).printWeights();
		}
	}
	
	
	
	
	
	
	
	
	
	
}
