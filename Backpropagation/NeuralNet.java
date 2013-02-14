package Backpropagation;

import java.util.Random;
import Main.Matrix;
import Main.SupervisedLearner;


public class NeuralNet extends SupervisedLearner {

	private NetworkManager _networkManager;
	private double _learningRate;
	private double _momentum;
	
	public NeuralNet(Random rand) {
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
		runBackPropagationPrediction(features, labels);
	}
	
	private void runBackPropagationTraining (Matrix inputs, Matrix targets) {
		// initialize the neural network to have as many input nodes as features and
		//	as many output units as there are output classes
		_networkManager.initializeNetwork(inputs.cols(), 2, 2, targets.cols());
		
		for (int i = 0; i < 3; i++) {
			System.out.println("Epoch " + (i+1) + "----------------");
			runAnEpoch(inputs, targets);
			_networkManager.printWeights();
			
		}
		
		/*boolean stoppingCriteriaMet = false;
		while (!stoppingCriteriaMet) {
			runAnEpoch (_networkManager, inputs, targets);
			stoppingCriteriaMet = stoppingCriteriaMet(_networkManager.getMSE());
		}*/
	}
	
	private void runBackPropagationPrediction (double[] inputs, double[] target) {
		
	}
	
	public void runAnEpoch (Matrix inputs, Matrix targets) {
		for (int i = 0; i < inputs.rows(); i++) {
			_networkManager.propagateInputForward(inputs.row(i), targets.row(i));
			_networkManager.propagateErrorsBack();
		}
	}
	
	public boolean stoppingCriteriaMet(double mse) {
		return true;
	}

}
