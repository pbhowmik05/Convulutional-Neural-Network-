public class LayerLinear extends Layer {
	
	LayerLinear(int inputs, int outputs) {
            super(inputs, outputs);
		// TODO Auto-generated constructor stub
	}
        protected double[] vals;
        protected int start;
        protected int len;
        
    @Override
    void activate(Vec weights, Vec x)
    {

        activation.fill(0);
        Vec b = new Vec(weights, 0, mOutput_size);
        Vec M = new Vec(weights, mOutput_size, mInput_size*mOutput_size);
        //System.out.println("weights values:  " +weights.toString());
        
        for(int r =0 ; r< mOutput_size; r++)           //edited here
        {           
            double value=0;
            for(int c = 0; c < mInput_size; c++)
            {
                //System.out.println("x values ::: " +x.get(c));
                value += x.get(c)*M.get(r*mInput_size + c);  
             //   System.out.println(" values:  " +value);
            }
            activation.set(r, value);
        } 
        
        activation.add(b);
        //System.out.println("Activation values:  " +activation.toString());
    }

    static void vecCrossMultiplyAdd(Vec x, Vec y, Matrix addTo)
    {       
        if( addTo.cols() != y.size() || addTo.rows() != x.size())
        {
            System.err.println("Size matching error: addTo.cols() != y.size() || addTo.rows() != x.size()");
            return;
        }
        
        for(int r = 0; r < x.size(); r++)            
        {
            Vec row = addTo.row(r);
            for(int c= 0; c< y.size(); c++)
            {
                double value = x.get(r)*y.get(c);
                row.set(c, value + row.get(c));
                
            }
        }    
    }
 
    @Override
    void backprop(Vec weights, Vec prevBlame) {
        
       // System.out.println("Traing backprop\n");
        Matrix M = new Matrix(0, mInput_size);
        for(int j=0; j<mOutput_size; j++) {
            Vec M_row = new Vec(mInput_size);
            for(int i=0; i<M_row.len; i++) {
                M_row.set(i, weights.get(j*mInput_size + mOutput_size + i));                
            }
                M.takeRow(M_row.vals);
        }       
        Matrix M_transpose = M.transpose();
        
        for(int i=0; i< M_transpose.rows(); i++) {       
            Vec row = M_transpose.row(i);            
            prevBlame.set(i, row.dotProduct(blame));   
            //System.out.println(prevBlame);
        }        
   
    }


    @Override
    void updateGradient(Vec x, Vec gradient) {
        Vec b = new Vec(mOutput_size);
        for(int i=0; i<b.len; i++) {
            //double k = blame.get(i+mOutput_size*mInput_size);
            //b.set(i+mOutput_size*mInput_size, k);
            double k = blame.get(i); //modified 
            b.set(i, k);
        }

        //Matrix gm = outer_product(blame, x);
        Matrix gm = new Matrix(0, mInput_size); //need to check        
        blame.crossMult(x, gm);
	for(int i = 0; i < mOutput_size; i++)
            gradient.set(i, b.get(i));
		
        for(int j = 0; j <gm.rows(); j++) {            
            for(int k = 0; k <gm.cols(); k++)
                gradient.set(k+j*gm.cols()+mOutput_size, gm.row(j).get(k));
        }        
    }
}
