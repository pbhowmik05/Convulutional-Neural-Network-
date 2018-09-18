abstract class Layer
{
	protected int mInput_size;
	protected int mOutput_size;	
        protected Vec activation;
	protected Vec blame;	


	Layer(int inputs, int outputs)
	{
		activation = new Vec(outputs);
		blame = new Vec(outputs);
		mInput_size = inputs;
		mOutput_size = outputs;

	}

	abstract void activate(Vec weights, Vec x);
	abstract void backprop(Vec weights, Vec prevBlame);
	abstract void updateGradient(Vec x, Vec gradient);

}