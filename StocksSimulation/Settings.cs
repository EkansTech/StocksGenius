using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public class SimSettings
    {
        public const float EffectivePredictionResult = 0.9F;

        public const float BuySellPenalty = 0.002F;

        public const float MinProfitRatio = 0.01F;

        public const float InvestmentMoney = 1000;
    };
}
