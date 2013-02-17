package Backpropagation;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
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
		_momentum = 0;
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
	}
	
	private void runBackPropagationTraining (Matrix inputs, Matrix targets) throws IOException {
		// initialize the neural network to have as many input nodes as features and
		//	as many output units as there are output classes
		_networkManager.initializeNetwork(inputs.cols(), 2*inputs.cols(), 1, targets.valueCount(0),
											inputs.rows());
		String sFileName = "LearningRateTest-" + Double.toString(_learningRate) + ".csv";
		FileWriter writer = new FileWriter(sFileName);
		double mse = 0;
		double prev_mse = 0;
		double classificationAccuracy = 0;
		int epochs = 0;
		boolean stoppingCriteriaMet = false;
		writer.append("Epochs, Percent Misclassification, Training MSE\n");
		for (int i = 0; i < 100; i++){
			prev_mse = _networkManager.getPrevNetworkMSE();
			runAnEpoch(inputs, targets);
			epochs++;
			System.out.println("Epochs: " + epochs);
			mse = _networkManager.getNetworkMSE();
			classificationAccuracy = _networkManager.getClassificationAccuracy();
			writer.append(Integer.toString(epochs));
			writer.append(",");
			writer.append(Double.toString(1 - classificationAccuracy));
			writer.append(",");
			writer.append(Double.toString(mse));
			writer.append("\n");
			writer.flush();
			stoppingCriteriaMet = stoppingCriteriaMet(prev_mse, mse);
		}
		writer.close();
	}
	
	public void runAnEpoch (Matrix inputs, Matrix targets) {
		for (int i = 0; i < inputs.rows(); i++) {
			_networkManager.propagateInputForward(inputs.row(i), targets.row(i));
			_networkManager.keepTrackOfOutputUnitErrorSum();
			_networkManager.propagateErrorsBack();
			_networkManager.updateAllWeights();
		}
	}
	
	public boolean stoppingCriteriaMet(double prevError, double error) {
		return Math.abs(prevError - error) < 0.001;
	}

}
