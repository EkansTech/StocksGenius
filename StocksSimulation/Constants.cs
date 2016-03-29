using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public static class Constants
    {
        public const int SimulationRange = 500;

        public const double EffectivePredictionResult = 0.65;

        public const double BuyActionPenalty = 1.002;

        public const double SellActionPenalty = 0.998;
    }
}
