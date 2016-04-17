using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StocksSimulation
{
    public static class Log
    {
        #region Enums

        public enum LogLevelType
        {
            Detailed,
            Important,
            Warning,
            Error
        }

        #endregion

        public struct LogMessage
        {
            public string String { get; set; }

            public LogLevelType LogLevel { get; set; }
        }

        #region Properties

        private static List<LogMessage> m_LogMessages = new List<LogMessage>();

        public static List<LogMessage> LogMessages
        {
            get { return m_LogMessages; }
            set { m_LogMessages = value; }
        }

        private static bool m_ConnectToConsole = true;

        public static bool ConnectToConsole
        {
            get { return m_ConnectToConsole; }
            set { m_ConnectToConsole = value; }
        }

        private static LogLevelType m_LogLevel = LogLevelType.Detailed;

        public static LogLevelType LogLevel
        {
            get { return m_LogLevel; }
            set { m_LogLevel = value; }
        }



        #endregion

        #region Interface

        public static void AddMessage(String format, params object[] args)
        {
            AddMessage(LogLevelType.Detailed, format, args);
        }

        public static void AddMessage(LogLevelType logLevel, String format, params object[] args)
        {
            m_LogMessages.Add(new LogMessage() { String = string.Format(format + Environment.NewLine, args), LogLevel = logLevel});
            if (m_ConnectToConsole)
            {
                if ((int)logLevel >= (int)m_LogLevel)
                {
                    Console.WriteLine(format, args);
                }
            }
        }

        public static string ToString(LogLevelType logLevel = LogLevelType.Detailed)
        {
            string logString = string.Empty;
            foreach (LogMessage logMessage in m_LogMessages)
            {
                if ((int)logLevel <= (int)LogLevelType.Detailed)
                {
                    logString += logMessage.String;
                }
            }

            return logString;
        }

        public static void SaveLogToFile(string filePath, LogLevelType logLevel = LogLevelType.Detailed)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                foreach (LogMessage logMessage in m_LogMessages)
                {
                    if ((int)logLevel <= (int)LogLevelType.Detailed)
                    {
                        writer.WriteLine(logMessage.String);
                    }
                }
            }
        }

        #endregion
    }
}
