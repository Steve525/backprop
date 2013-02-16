package Backpropagation;

import java.util.ArrayList;
import java.util.Random;
import Main.Matrix;
import Main.SupervisedLearner;


public class NeuralNet extends SupervisedLearner {

	private NetworkManager _networkManager;
	private double _learningRate;
	private double _momentum;
	private ArrayList<Double> _labels;
	
	public NeuralNet(Random rand) {
		_labels = new ArrayList<Double>();
		_learningRate = 0.1;
		_momentum = 0;
		_networkManager = new NetworkManager(_learningRate, _momentum);
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		runBackPropagationTraining(features, labels);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = _networkManager.getNetworkOutput(features);
		//System.out.println(labels[0]);
	}
	
	private void runBackPropagationTraining (Matrix inputs, Matrix targets) {
		// initialize the neural network to have as many input nodes as features and
		//	as many output units as there are output classes
		_networkManager.initializeNetwork(inputs.cols(), 2*inputs.cols(), 1, targets.valueCount(0));
		
		for (int i = 0; i < 200; i++) {
			//System.out.println("Epoch " + (i+1) + "----------------");
			//_networkManager.printWeights();
			runAnEpoch(inputs, targets);
		}
		
		/*boolean stoppingCriteriaMet = false;
		while (!stoppingCriteriaMet) {
			runAnEpoch (_networkManager, inputs, targets);
			stoppingCriteriaMet = stoppingCriteriaMet(_networkManager.getMSE());
		}*/
	}
	
	private void runBackPropagationPrediction (double[] inputs, double[] answer) {
		
	}
	
	public void runAnEpoch (Matrix inputs, Matrix targets) {
		for (int i = 0; i < inputs.rows(); i++) {
			_networkManager.propagateInputForward(inputs.row(i), targets.row(i));
			//_networkManager.printInputOutput(inputs.row(i), targets.row(i));
			//System.out.println("Forward propagating...");
			//_networkManager.printPredictedOutput();
			//System.out.println("Back propagating...");
			_networkManager.propagateErrorsBack();
			//_networkManager.printErrorTerms();
			//System.out.println("Descending gradient...");
			_networkManager.updateAllWeights();
		}
	}
	
	public boolean stoppingCriteriaMet(double mse) {
		return true;
	}

}
