using System;
using System.Text;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace CSErrorModel
{
    public class GaussianMixture : MixtureEstimator<Gaussian>, CanGetLogProb<double>, Sampleable<double>
    {
        public double GetMode()
        {
            double mode = double.NaN;
            double logp = double.NegativeInfinity;
            for (int i = 0; i < components.Count; i++)
            {
                double modeLocal = GetMode(components[i].GetMean());
                double logpLocal = GetLogProb(modeLocal);
                if (logpLocal > logp)
                {
                    logp = logpLocal;
                    mode = modeLocal;
                }
            }
            return mode;
        }

        private double GetMode(double x0)
        {
            // logp = log p(x) = log sum wi*N(x|mi,vi)
            // dlogp = 0 ==> sum wi*N(x|mi,vi)*(x-mi)/vi = 0 ==> x = [sum wi*N(x|mi,vi)*mi/vi] / [sum wi*N(x|mi,vi)/vi]
            double x = x0;
            int iter = 0;
            while (iter == 0 || (Math.Abs(x0 - x) > 1e-6) && iter < 10)
            {
                double numerator = 0, denominator = 0;
                for (int i = 0; i < components.Count; i++)
                {
                    double p = weights[i] * Math.Exp(components[i].GetLogProb(x));
                    numerator += p * components[i].MeanTimesPrecision;
                    denominator += p * components[i].Precision;
                }
                x0 = x;
                x = numerator / denominator;
                iter++;
            }
            return x;
        }

        public double GetLogProb(double x)
        {
            double logProb = double.NegativeInfinity;
            double weightSum = 0;
            for (int i = 0; i < weights.Count; i++)
            {
                weightSum += weights[i];
                logProb = MMath.LogSumExp(logProb, Math.Log(weights[i]) + components[i].GetLogProb(x));
            }
            return logProb - Math.Log(weightSum);
        }

        public double Sample(double result) => Sample();
        public double Sample()
        {
            int i = Rand.Sample(weights, WeightSum());
            return components[i].Sample();
        }

        //public void SetToProductWith(Gaussian dist)
        //{
        //    for (int i = 0; i < components.Count; i++) components[i] *= dist;
        //}

        public void SetToProductWith(Gaussian dist)
        {
            for (int i = 0; i < components.Count; i++)
            {
                components[i] *= dist;
                double logw = Math.Log(weights[i]) + components[i].GetLogAverageOf(dist);
                weights[i] = Math.Exp(logw);
            }
            Normalize();
        }

        public static GaussianMixture operator *(Gaussian a, GaussianMixture b) => b * a;
        public static GaussianMixture operator *(GaussianMixture a, Gaussian b)
        {
            var result = new GaussianMixture();
            foreach (var component in a.Components) result.Add(component * b);
            return result;
        }

        public static GaussianMixture operator *(GaussianMixture a, GaussianMixture b)
        {
            var result = new GaussianMixture();
            for (int i = 0; i < a.Components.Count; i++)
            {
                for (int j = 0; j < b.Components.Count; j++)
                {
                    result.Add(a.Components[i] * b.Components[j], a.Weights[i] * b.Weights[j]);
                }
            }
            result.Normalize();
            return result;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < components.Count; i++) sb.Append($"\n{weights[i]}, {components[i]}");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Linear Gaussian mixture: y ~ sum_i { w_i * N(y | a_i * x + b_i, v_i) }
    /// </summary>
    public static class LinearGaussianMixture
    {
        // Returns the Gaussian mixture distribution over y given x.
        public static GaussianMixture Forward(double x, Vector weights, double[] a, double[] b, double[] tau)
        {
            if (weights.Count != a.Length || weights.Count != b.Length || weights.Count != tau.Length)
                throw new ArgumentException("Inconsistent number of components in linear Gaussian mixture.");

            var result = new GaussianMixture();
            for (int i = 0; i < weights.Count; i++)
            {
                result.Add(Gaussian.FromMeanAndPrecision(a[i] * x + b[i], tau[i]), weights[i]);
            }
            result.Normalize();

            return result;
        }

        // Returns the Gaussian mixture distribution over x given y.
        public static GaussianMixture Backward(double y, Vector weights, double[] a, double[] b, double[] tau)
        {
            if (weights.Count != a.Length || weights.Count != b.Length || weights.Count != tau.Length)
                throw new ArgumentException("Inconsistent number of components in linear Gaussian mixture.");

            var result = new GaussianMixture();
            for (int i = 0; i < weights.Count; i++)
            {
                result.Add(Gaussian.FromMeanAndPrecision((y - b[i]) / a[i], a[i] * a[i] * tau[i]), weights[i] / a[i]);
            }
            result.Normalize();

            return result;
        }
    }
}