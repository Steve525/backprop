package Backpropagation;

import java.util.ArrayList;

public class NetworkManager {
	
	private ArrayList<INode> _inputLayer;
	private ArrayList<ArrayList<INode>> _hiddenLayers;
	private ArrayList<INode> _outputLayer;
	private double _learningRate;
	private double _momentum;
	private double _networkMSE;
	private int _unitCount;
	private int _iterations;
	private double _maxWeightRange;
	private double _minWeightRange;
	private int _correctlyClassified;
	private double _networkOutput;
	private double _numberOfInstances;

	public NetworkManager(double learningRate, double momentum, double maxWeightRange, double minWeightRange) {
		_unitCount = 0;
		_learningRate = learningRate;
		_momentum = momentum;
		_networkMSE = 0;
		_inputLayer = new ArrayList<INode>();
		_hiddenLayers = new ArrayList<ArrayList<INode>>();
		_outputLayer = new ArrayList<INode>();
		_iterations = 0;
		_maxWeightRange = maxWeightRange;
		_minWeightRange = minWeightRange;
		_correctlyClassified = 0;
		_networkOutput = 0;
	}
	
	public void initializeNetwork(int n_in, int n_hidden, int n_hiddenLayers, int n_output,
																double numberOfInstances) {
		
		_numberOfInstances = numberOfInstances;
		
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
		
		// Initialize weights of all layers to a random value between
		//	(variable max:0.05 and min:-0.05)
		initializeWeights (_maxWeightRange, _minWeightRange);
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
		
		// 0.1, 0.2, -0.1,                                                      
	    // -0.2, 0.3, -0.3, 
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
		
		// 0.1, -0.2, -0.3, 
	    // 0.2, -0.1, 0.3,
		INode hiddenUnit3 = new HiddenUnit(3);
		((AbstractUnit) hiddenUnit3).setBiasWeight(0.1);
		((AbstractUnit) hiddenUnit3).setWeightMap(hiddenUnit1, -0.2);
		((AbstractUnit) hiddenUnit3).setWeightMap(hiddenUnit2, -0.3);
		INode hiddenUnit4 = new HiddenUnit(4);
		((AbstractUnit) hiddenUnit4).setBiasWeight(0.2);
		((AbstractUnit) hiddenUnit4).setWeightMap(hiddenUnit1, -0.1);
		((AbstractUnit) hiddenUnit4).setWeightMap(hiddenUnit2, 0.3);
		_hiddenLayers.get(1).add(hiddenUnit3);
		_hiddenLayers.get(1).add(hiddenUnit4);
		
		// 0.2, -0.1, 0.3, 
	    // 0.1, -0.2, -0.3
		INode outputUnit1 = new OutputUnit(5);
		((AbstractUnit) outputUnit1).setBiasWeight(0.2);
		((AbstractUnit) outputUnit1).setWeightMap(hiddenUnit3, -0.1);
		((AbstractUnit) outputUnit1).setWeightMap(hiddenUnit4, 0.3);
		INode outputUnit2 = new OutputUnit(6);
		((AbstractUnit) outputUnit2).setBiasWeight(0.1);
		((AbstractUnit) outputUnit2).setWeightMap(hiddenUnit3, -0.2);
		((AbstractUnit) outputUnit2).setWeightMap(hiddenUnit4, -0.3);
		_outputLayer.add(outputUnit1);
		_outputLayer.add(outputUnit2);
	}
	
	public double getClassificationAccuracy() {
		double classificationAccuracy = (_correctlyClassified / _numberOfInstances);
		_correctlyClassified = 0;
		return classificationAccuracy;
	}
	
	public double getCorrectlyClassified() { return _correctlyClassified; }
	
	public double getPrevNetworkMSE() {
		double prevError = (_networkMSE / _numberOfInstances);	// save MSE
		_networkMSE = 0;	// reset MSE
		return prevError;
	}
	
	public double getNetworkMSE() {
		return (_networkMSE / _numberOfInstances);
	}
	
	public void keepTrackOfOutputUnitErrorSum() {
		_networkMSE += getAllOutputUnitError();
	}
	
	private double getAllOutputUnitError() {
		double error = 0;
		for (int i = 0; i < _outputLayer.size(); i++) {
			OutputUnit outputUnit = (OutputUnit) _outputLayer.get(i);
			error += Math.pow(outputUnit.getTarget() - outputUnit.getOutput(), 2);
		}
		return error;
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
		double output = (double) getHighestOutputUnitID();
		if (output == target[0])
			_correctlyClassified++;
	}
	
	public double getNetworkOutput (double[] inputs) {
		giveInputNodesInput(inputs);
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
		return (double) getHighestOutputUnitID();
	}
	
	public void printHiddenUnitWeights () {
		System.out.print(((AbstractUnit) _hiddenLayers.get(0).get(1)).toStringWeights());
	}
	
	private int getHighestOutputUnitID () {
		int unitID = -1;
		double highestNet = -10000000;
		double net = 0;
		//System.out.println("OutputLayer Size: " + _outputLayer.size());
		for (int i = 0; i < _outputLayer.size(); i++) {
			net = ((OutputUnit) _outputLayer.get(i)).getNet();
			//System.out.println("Output 1 Net: " + net);
			if (net > highestNet) {
				highestNet = net;
				unitID = i;
			}
		}
		//System.out.println("unitID: " + unitID);
		return unitID;
	}
	
	private void giveInputNodesInput (double[] inputs) {
		for (int i = 0; i < _inputLayer.size(); i++) {
			InputNode inputNode = (InputNode) _inputLayer.get(i);
			inputNode.setInput(inputs[i]);
		}
	}
	
	private void giveOutputUnitsTargets (double[] target) {
		//System.out.println("		KJSNKJSNK: " + target[0]);
		//	For the iris.arff data file:
		//	If target is a 2, then a corresponding {0, 0, 1} will be passed to the outputUnits.
		//	If target is a 1, then {0, 1, 0} is passed and so forth.
		for (int i = 0; i < _outputLayer.size(); i++) {
			OutputUnit outputUnit = (OutputUnit) _outputLayer.get(i);
			if (target[0] == i) // if target is a 2, when i is 2, set third outputUnit target to '1' (iris.arff)
				outputUnit.setTarget(1);
			else
				outputUnit.setTarget(0);
		}
	}
	
	public void propagateErrorsBack() {
		calculateOutputUnitErrors();
		calculateHiddenUnitErrors();
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
	
	public void updateAllWeights() {
		for (int i = 0; i < _hiddenLayers.size(); i++) {
			ArrayList<INode> hiddenLayer = _hiddenLayers.get(i);
			for (int j = 0; j < hiddenLayer.size(); j++) {
				((AbstractUnit) hiddenLayer.get(j)).updateWeights(_learningRate, _momentum, _iterations);
			}
		}
		for (int i = 0; i < _outputLayer.size(); i++) {
			((AbstractUnit) _outputLayer.get(i)).updateWeights(_learningRate, _momentum, _iterations);
		}
		_iterations++;
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
				outputUnit.setDeltaWeightMap(hiddenUnit, 0);
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
					hiddenUnit.setDeltaWeightMap(nextHiddenUnit, 0);
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
				hiddenUnit.setDeltaWeightMap(inputNode, 0);
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
	
	public void printInputOutput(double[] inputs, double[] targets) {
		System.out.print("Input vector: ");
		for (int i = 0; i < inputs.length; i++) {
			System.out.print(_inputLayer.get(i).getOutput() + ", ");
		}
		System.out.println();
		System.out.print("Target output: ");
		for (int i = 0; i < targets.length; i++) {
			System.out.print(((OutputUnit) _outputLayer.get(i)).getTarget() + ", ");
		}
		System.out.println();
	}
	
	public void printPredictedOutput() {
		System.out.print("Predicted Output: ");
		for (int i = 0; i < _outputLayer.size(); i++) {
			System.out.print(_outputLayer.get(i).getOutput() + ", ");
		}
		System.out.println();
	}
	
	public void printErrorTerms() {
		System.out.println("Error terms: ");
		for (int i = 0; i < _hiddenLayers.size(); i++) {
			ArrayList<INode> hiddenLayer = _hiddenLayers.get(i);
			for (int j = 0; j < hiddenLayer.size(); j++) {
				((AbstractUnit) hiddenLayer.get(j)).printErrorTerm();
			}
			System.out.println();
		}
		for (int i = 0; i < _outputLayer.size(); i++) {
			((AbstractUnit) _outputLayer.get(i)).printErrorTerm();
		}
		System.out.println();
	}
}
