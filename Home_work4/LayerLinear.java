import java.util.Random;
public class LayerLinear extends Layer
{
    LayerLinear(int inputs, int outputs)
    {
        super(inputs, outputs);
    }
	protected double[] vals;
	protected int start;
        protected int len;
        
    public int size() {
        return len;
    }    
    
    @Override
    void activate(Vec weights, Vec x)
    {
        //weights = new Vec(new double[]{0.1,0.2,0.3,0.4,0.1,0.2,0.3});
       // Vec M = new Vec(new double[]{0.1,0.2,0.3,0.4});
        //Vec b = new Vec(new double[]{0.1,0.2,0.3});
        activation.fill(0);
        Vec M = new Vec(weights, mOutput_size, mInput_size*mOutput_size);
        Vec b = new Vec(weights, 0, mOutput_size);

        
        // System.out.println("activation function done: " +activation);
        for(int r =0 ; r< mOutput_size; r++)           //edited here
        {
            double value = 0;
            for(int c = 0; c < mInput_size; c++)
            {
                value += x.get(c)*M.get(r*mInput_size + c);    
            }
            activation.set(r, value);
        } 
        activation.add(b);
    } 
  

/*    public void crossMult(Vec x, Matrix dummy) {
        for(int i = 0; i < size(); i++) {
            Vec temp = new Vec(0);
            temp.copy(x);
            temp.scale(get(i));
            dummy.takeRow(temp.vals);
        }
    }*/
    
    
    private static Matrix outer_product(Vec x, Vec y)
    {
        Matrix ret = new Matrix(0, y.size());

        for (int r= 0; r<x.size(); r++)
        {
            double [] products =new double[y.size()];
            for(int c=0; c<y.size(); c++)
            {
                products[c] = x.get(r)*y.get(c);
            }
            ret.takeRow(products);
        }
        return ret;
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
       
    private static Vec getColumnwiseMean(Matrix m)
    {
        Vec v = new Vec(m.cols());
        for (int i =0; i< m.cols(); i++)
        {
            v.set(i,m.columnMean(i));            
        }
        return v;   
    }
    public double get(int index)
    {
	return vals[start + index];
    }

    void ordinary_least_squares(Matrix X, Matrix Y, Vec weights)
    {
        Matrix y_yMean_cross_x_xMean = new Matrix(mOutput_size, mInput_size);
        Matrix x_xMean_cross_x_xMean = new Matrix(mInput_size, mInput_size);
        
        Vec xMean = getColumnwiseMean(X);
        Vec yMean = getColumnwiseMean(Y);
        
        for (int i=0; i<X.rows(); i++)
        {
            Vec row_x = new Vec(0);
            row_x.copy(X.row(i));
            Vec row_y = new Vec(0);
            row_y.copy(Y.row(i));
            
            // Calculate Means first, then subtract from the original value
            for(int c = 0; c<row_x.size(); c++)
            {
                row_x.set(c, row_x.get(c)- xMean.get(c));
            }
            
            for(int c=0; c<row_y.size(); c++)
            {
                row_y.set(c, row_y.get(c));
            }
            
            vecCrossMultiplyAdd(row_y, row_x, y_yMean_cross_x_xMean);
            vecCrossMultiplyAdd(row_x, row_x, x_xMean_cross_x_xMean);
        }
        
        x_xMean_cross_x_xMean = x_xMean_cross_x_xMean.pseudoInverse();
        
        Matrix M = Matrix.multiply(y_yMean_cross_x_xMean, x_xMean_cross_x_xMean, false, false);
        //System.out.println("M-->\n"+M.toString());
        
        Vec B = new Vec(mOutput_size);
        
        for(int i = 0; i< mOutput_size; i++)
        {
            Vec row = M.row(i);
            B.set(i, yMean.get(i) - row.dotProduct(xMean));
        }

        Vec bWeights = new Vec(weights, 0, B.size());
        bWeights.add(B);
        

        for(int i = 0; i<M.rows(); i++)
        {
            for(int j = 0; j < M.cols(); j++)
            {       
                weights.set(i*M.cols()+j+mOutput_size, M.row(i).get(j));
            }
        }
 
    } 
    


    @Override
    void backProp(Vec weights, Vec prevBlame) {
        
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