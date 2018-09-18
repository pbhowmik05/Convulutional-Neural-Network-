public class LayerConv extends Layer{
/*	LayerConv(int inputs, int outputs) {
		super(inputs, outputs);
	} */

    int [] mInputs;
    int [] mFilters;
    int [] mOutputs;
    int FilterCount;
    
//    int inputDims;
//    int filterDims;
//    int outputDims;
//    int num_filter;


    static int getElements(int[] array) {
	int elements = 1;
        //   System.out.println("array.length         "+array.length);
	for(int i = 0; i < array.length; i++){
            elements *= array[i];
                        //System.out.println("element         "+elements);
        }
               // System.out.println("element         "+elements);
	return elements;
    }        
      //  int outputSZ =  getElements(mOutputs);
      //  int filterSZ =  getElements(mFilters);
        
    LayerConv(int[] inputDims, int[] filterDims, int[] outputDims, int num_filter) {
        super(getElements(inputDims), getElements(outputDims)*num_filter);
        mInputs = inputDims;
        mFilters = filterDims;
        mOutputs = outputDims;
        this.FilterCount = num_filter;            
    }
    
    @Override
    void activate(Vec weights, Vec x)
    {
        
        int filterSZ =  getElements(mFilters);
        int outputSZ =  getElements(mOutputs);
        
        int z = outputSZ*FilterCount;
        Vec outConv = new Vec(z);

        for(int r =0 ; r< FilterCount; r++)  {         
            Vec temp = new Vec(outputSZ);
            Vec Vec_filter = new Vec(filterSZ);
            
            Tensor Vec_x = new Tensor(x, mInputs);
            
            Tensor Vec_out = new Tensor(temp, mOutputs);
            
            for (int i = 0; i < filterSZ; i++){
                int dim = r*(filterSZ+1)+i;
               // System.out.println("dim    :"+dim);
                Vec_filter.set(i, weights.get(dim));   //bujhi nai
                //System.out.println("Vec_filter       : \n"+Vec_filter.toString());
            }
            Tensor Ten_filter = new Tensor(Vec_filter, mFilters);
            Tensor.convolve(Vec_x, Ten_filter, Vec_out, false, 1);
            
            for (int j = 0; j < outputSZ; j++){
                outConv.set(r*outputSZ+j, Vec_out.get(j)+weights.get(r*(filterSZ+1))); //bujhi nai
            }
        } 
        activation.copy(outConv);
    }
    
   
    @Override
    void backProp(Vec weights, Vec prevBlame) {
            int outputSZ =  getElements(mOutputs);
            int filterSZ =  getElements(mFilters);
            Vec backVec = new Vec(prevBlame.len);
            
            Tensor backTen = new Tensor(backVec, mInputs);
            for (int i = 0; i < FilterCount; i++){
                Vec Vec_weight = new Vec(filterSZ);
                for (int j = 0; j < filterSZ; j++){
                    Vec_weight.set(j, weights.get(i*(filterSZ+1)+j));   
          //        System.out.println("Vec_weightVec_weight :"+Vec_weight);
                }
                Tensor backfilter = new Tensor(Vec_weight, mFilters);
                Vec Vec_blame = new Vec(blame, i*outputSZ, outputSZ);
                Tensor backblame = new Tensor(Vec_blame,mOutputs);
                
                Tensor.convolve(backfilter, backblame, backTen, true, 1);
                
                for(int j = 0; j < prevBlame.len; j++)
                    prevBlame.set(j, backTen.get(j));
            }
        }
        
    @Override
    void updateGradient(Vec x, Vec gradient) {
        int filterSZ =  getElements(mFilters);
        int outputSZ =  getElements(mOutputs);

        for(int i = 0; i < FilterCount; i++) {
            Tensor gradIn = new Tensor(x, mInputs);
            Vec gradVecblame = new Vec(blame,i*outputSZ, outputSZ);
            Tensor gradBlame = new Tensor(gradVecblame, mOutputs);
            Vec gradVecFilter = new Vec(filterSZ);
            Tensor gradOut = new Tensor(gradVecFilter, mFilters);
            
            Tensor.convolve(gradIn, gradBlame, gradOut, false, 1);
            
            Vec gradOutput = new Vec(gradOut.vals);
            Vec Blame_grad = new Vec(blame, i*outputSZ, outputSZ);
            double sum = 0; int len = Blame_grad.len;
            for(int j = 0; j < len; j++){
                sum +=Blame_grad.vals[j];
            //    System.out.println("print sum :"+sum);
            }
            
            Vec biasVector = new Vec(new double[]{sum});
            gradient.set(i*gradOutput.len+i, biasVector.get(0));

            int indexVal = (gradOutput.len * i) + i;
            for(int j = 0; j < gradOutput.len; j++){
                gradient.set(indexVal, gradOutput.vals[j]);
                indexVal++;
            }
        }            
    } 
}