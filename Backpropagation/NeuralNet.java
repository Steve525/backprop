package Backpropagation;

import java.util.Random;
import Main.Matrix;
import Main.SupervisedLearner;


public class NeuralNet extends SupervisedLearner {

	private NetworkManager _networkManager;
	private double _learningRate;
	private double _momentum;
	private double _maxWeightRange;
	private double _minWeightRange;
	
	public NeuralNet(Random rand) {
		_learningRate = 0.1;
		_momentum = 0.9;
		_maxWeightRange = 0.05;
		_minWeightRange = -0.05;
		_networkManager = new NetworkManager(_learningRate, _momentum, _maxWeightRange, _minWeightRange);
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
		double error = 0;
		double prevError = 0;
		int epochs = 0;
		boolean stoppingCriteriaMet = false;
		while (!stoppingCriteriaMet) {
			prevError = _networkManager.getPrevNetworkMSE();
			error = runAnEpoch(inputs, targets);
			epochs++;
			System.out.println ("prevError: " + prevError + " error: " + error);
			stoppingCriteriaMet = stoppingCriteriaMet(prevError, error);
		}
		System.out.println("Epochs: " + epochs);
	}
	
	public double runAnEpoch (Matrix inputs, Matrix targets) {
		for (int i = 0; i < inputs.rows(); i++) {
			_networkManager.propagateInputForward(inputs.row(i), targets.row(i));
			_networkManager.keepTrackOfOutputUnitErrorSum();
			//_networkManager.printInputOutput(inputs.row(i), targets.row(i));
			//System.out.println("Forward propagating...");
			//_networkManager.printPredictedOutput();
			//System.out.println("Back propagating...");
			_networkManager.propagateErrorsBack();
			//_networkManager.printErrorTerms();
			//System.out.println("Descending gradient...");
			_networkManager.updateAllWeights();
		}
		double error = _networkManager.getNetworkMSE();
		return error;
	}
	
	public boolean stoppingCriteriaMet(double prevError, double error) {
		return Math.abs(prevError - error) < 0.01;
	}

}
