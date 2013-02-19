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
		_learningRate = 0.2;
		_momentum = 0;
		_maxWeightRange = 0.05;
		_minWeightRange = -0.05;
		_networkManager = new NetworkManager(_learningRate, _momentum, _maxWeightRange, _minWeightRange);
	}

	@Override
	public void train(Matrix features, Matrix labels, Matrix vFeatures, Matrix vLabels, Matrix confusion) throws Exception {
		runBackPropagationTraining(features, labels, vFeatures, vLabels, confusion);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = _networkManager.getNetworkOutput(features);
	}
	
	private void runBackPropagationTraining (Matrix inputs, Matrix targets, Matrix vFeatures, Matrix vLabels, Matrix confusion) throws Exception {
		// initialize the neural network to have as many input nodes as features and
		//	as many output units as there are output classes
		int n_input = inputs.cols();
		int n_hidden = 3;
		int n_hiddenLayers = 1;
		int n_output = targets.valueCount(0);
		
		_networkManager.initializeNetwork(n_input, n_hidden, n_hiddenLayers, n_output,
											inputs.rows());
		System.out.print("input, hidden, hidden layers, output\n");
		System.out.print(n_input);
		System.out.print(",");
		System.out.print(n_hidden);
		System.out.print(",");
		System.out.print(n_hiddenLayers);
		System.out.print(",");
		System.out.print(n_output);
		System.out.print("\n");
		//String sFileName = "LearningRateTest-" + Double.toString(_learningRate) + 
							//"-" + n_hidden + "-layers-" + n_hiddenLayers + ".csv";
		//FileWriter writer = new FileWriter(sFileName);
		//System.out.println(sFileName);
		double trainingMSE = 0;
		double prev_mse = 0;
		double validationClassificationAccuracy = 0;
		double classificationAccuracy = 0;
		int epochs = 0;
		boolean stoppingCriteriaMet = false;
		System.out.print("Epochs, Training MSE, Validation Classification Accuracy\n");
		//writer.append("Epochs, Classification Accuracy, Training MSE\n");
		for (int i = 0; i < 2500; i++){
		//while (!stoppingCriteriaMet) {
			prev_mse = _networkManager.getPrevNetworkMSE();
			runAnEpoch(inputs, targets);
			epochs++;
			trainingMSE = _networkManager.getNetworkMSE();
			validationClassificationAccuracy = super.measureAccuracy(vFeatures, vLabels, confusion);
			classificationAccuracy = _networkManager.getClassificationAccuracy();
			System.out.print(epochs);
			System.out.print(",");
			_networkManager.printHiddenUnitWeights();
			//System.out.print(classificationAccuracy);
			//System.out.print(",");
			//System.out.print(trainingMSE);
			//System.out.print(",");
			//System.out.print(1 - validationClassificationAccuracy);
			//System.out.print("\n");
			/*writer.append(Integer.toString(epochs));
			writer.append(",");
			writer.append(Double.toString(classificationAccuracy));
			writer.append(",");
			writer.append(Double.toString(mse));
			writer.append("\n");
			writer.flush();*/
			stoppingCriteriaMet = stoppingCriteriaMet(prev_mse, trainingMSE);
		}
		//System.out.println("Epochs: " + epochs);
		//writer.close();
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
