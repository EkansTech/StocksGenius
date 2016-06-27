using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace StocksData
{
    /// <summary>
    /// Create a New INI file to store or load data
    /// </summary>
    public class IniFile
    {
        private Dictionary<string, Dictionary<string, string>> m_IniData = new Dictionary<string, Dictionary<string, string>>();

        private StreamWriter m_Writer;

        public string path;

        [DllImport("kernel32")]
        private static extern long WritePrivateProfileString(string section,
            string key, string val, string filePath);
        [DllImport("kernel32")]
        private static extern int GetPrivateProfileString(string section,
                 string key, string def, StringBuilder retVal,
            int size, string filePath);

        #region Constructors

        /// <summary>
        /// INIFile Constructor.
        /// </summary>
        /// <PARAM name="INIPath"></PARAM>
        public IniFile(string INIPath)
        {
            path = INIPath;
            string iniContent = new StreamReader(INIPath).ReadToEnd();
            string[] lines = iniContent.Split('\r', '\n');
            string section = string.Empty;

            foreach (string iniLine in lines)
            {
                string line = iniLine.Trim();

                if (string.IsNullOrWhiteSpace(line) || line.StartsWith(";"))
                {
                    continue;
                }

                if (line.StartsWith("["))
                {
                    section = line.Trim('[', ']');
                    m_IniData.Add(section, new Dictionary<string, string>());
                    continue;
                }

                if (!line.Contains("="))
                {
                    continue;
                }

                string key = line.Split('=')[0].Trim();
                string value = line.Split('=')[1].Trim();
                m_IniData[section].Add(key, value);
            }

            //m_Writer = new StreamWriter(INIPath);
        }

        ~IniFile()
        {
           // m_Writer.Close();
        }

        #endregion

        #region Interface

        /// <summary>
        /// Write Data to the INI File
        /// </summary>
        /// <PARAM name="Section"></PARAM>
        /// Section name
        /// <PARAM name="Key"></PARAM>
        /// Key Name
        /// <PARAM name="Value"></PARAM>
        /// Value Name
        public void IniWriteValue(string Section, string Key, string Value)
        {
            WritePrivateProfileString(Section, Key, Value, this.path);
        }

        public void IniWriteValue(string Section, string Key, double Value)
        {
            WritePrivateProfileString(Section, Key, Value.ToString(), this.path);
        }
        public void IniWriteValue(string Section, string Key, int Value)
        {
            WritePrivateProfileString(Section, Key, Value.ToString(), this.path);
        }
        public void IniWriteValue(string Section, string Key, DateTime Value)
        {
            WritePrivateProfileString(Section, Key, Value.ToShortDateString(), this.path);
        }

        ///// <summary>
        ///// Read Data Value From the Ini File
        ///// </summary>
        ///// <PARAM name="Section"></PARAM>
        ///// <PARAM name="Key"></PARAM>
        ///// <PARAM name="Path"></PARAM>
        ///// <returns></returns>
        //public string IniReadValue(string Section, string Key)
        //{
        //    StringBuilder temp = new StringBuilder(255);
        //    int i = GetPrivateProfileString(Section, Key, "", temp,
        //                                    255, this.path);
        //    return temp.ToString();

        //}

        public string IniReadValue(string section, string key)
        {
            if (!m_IniData.ContainsKey(section) || !m_IniData[section].ContainsKey(key))
            {
                return string.Empty;
            }
            else
            {
                return m_IniData[section][key];
            }
        }
        public double IniReadDoubleValue(string Section, string Key)
        {
            return Convert.ToDouble(IniReadValue(Section, Key));
        }
        public void IniReadDoubleValue(string Section, string Key, ref double value)
        {
            string valueString = IniReadValue(Section, Key);
            if (!string.IsNullOrWhiteSpace(valueString))
            {
                value = Convert.ToDouble(value);
            }
        }
        public int IniReadIntValue(string Section, string Key)
        {
            return Convert.ToInt32(IniReadValue(Section, Key));
        }
        public bool IniReadBoolValue(string Section, string Key)
        {
            return Convert.ToBoolean(IniReadValue(Section, Key));
        }
        public byte IniReadByteValue(string Section, string Key)
        {
            return Convert.ToByte(IniReadValue(Section, Key));
        }
        public DateTime IniReadDateTime(string Section, string Key)
        {
            return Convert.ToDateTime(IniReadValue(Section, Key));
        }

        public void IniReadDateTime(string Section, string Key, ref DateTime date)
        {
            string value = IniReadValue(Section, Key);
            if (!string.IsNullOrWhiteSpace(value))
            {
                date = Convert.ToDateTime(value);
            }
        }

        public void IniReadEnum<EnumType>(string Section, string Key, ref EnumType enumValue)
        {
            string value = IniReadValue(Section, Key);
            if (!string.IsNullOrWhiteSpace(value))
            {
                enumValue = (EnumType)Enum.Parse(typeof(EnumType), value);
            }
        }

        #endregion

        #region Private Methods

        #endregion
    }
}