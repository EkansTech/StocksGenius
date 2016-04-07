using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksData
{
    public struct CombinationItem
    {
        #region Properties

        public byte Range;
        public DataItem DataItem;

        #endregion

        #region Constructors

        public CombinationItem(byte range, DataItem combination)
        {
            Range = range;
            DataItem = combination;
        }

        public CombinationItem(string combinationItemString)
        {
            string[] stringParts = combinationItemString.Split('-');
            Range = Convert.ToByte(stringParts[0]);
            DataItem = (DataItem)Enum.Parse(typeof(DataItem), stringParts[1]);
        }

        #endregion

        #region Interface

        public bool Is(DataItem dataItem, int range)
        {
            return range == Range && dataItem == DataItem;
        }

        public ulong ToULong()
        {
            return ((ulong)1) << (DSSettings.ChangeItemsMap[this]);
        }

        static public CombinationItem Item(DataItem dataItem, byte range)
        {
            return new CombinationItem(range, dataItem);
        }

        static public List<CombinationItem> ULongToCombinationItems(ulong combination)
        {
            List<CombinationItem> combinationItems = new List<CombinationItem>();
            ulong combinationItem = 1;
            for (int combinationItemNum = 0; combinationItemNum < DSSettings.ChangeItems.Count; combinationItemNum++)
            {
                if ((combination & combinationItem) != 0)
                {
                    combinationItems.Add(DSSettings.ULongToCombinationItemMap[combinationItem]);
                }
                combinationItem *= 2;
            }

            return combinationItems;
        }

        static public ulong CombinationItemsToULong(List<CombinationItem> combinationTimes)
        {
            ulong combination = 0;
            foreach (CombinationItem combinationItem in combinationTimes)
            {
                combination |= combinationItem.ToULong();
            }

            return combination;
        }

        static public string CombinationToString(ulong combination)
        {
            if (combination == 0)
            {
                return string.Empty;
            }

            List<CombinationItem> combinationItems = ULongToCombinationItems(combination);

            string combinationString = combinationItems[0].ToString();

            for (int i = 1; i < combinationItems.Count; i++)
            {
                combinationString += "+" + combinationItems[i].ToString();
            }

            return combinationString;
        }

        static public List<CombinationItem> StringToItems(string combinationString)
        {
            string[] combinationItemsStrings = combinationString.Split('+');

            if (combinationItemsStrings.Length == 0)
            {
                return null;
            }

            List<CombinationItem> combinationItems = new List<CombinationItem>();

            foreach (string combinationItemString in combinationItemsStrings)
            {
                combinationItems.Add(new CombinationItem(combinationItemString));
            }

            return combinationItems;
        }

        static public ulong StringToCombinationULong(string combinationString)
        {
            string[] combinationItemsStrings = combinationString.Split('+');

            if (combinationItemsStrings.Length == 0)
            {
                return 0;
            }

            ulong combination = 0;

            foreach (string combinationItemString in combinationItemsStrings)
            {
                combination |= (new CombinationItem(combinationItemString)).ToULong();
            }

            return combination;
        }

        public override string ToString()
        {
            return Range.ToString() + "-" + DataItem.ToString();
        }

        #endregion
    }
}
