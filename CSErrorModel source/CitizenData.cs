using System.IO;
using System.Collections.Generic;

namespace CSErrorModel
{
    public class CitizenData
    {
        public int[] CitizenIDs { get; set; }
        public int[] TrueIndex { get; set; }
        public int[][] TrueIndexJagged { get; set; }
        public int[] Error { get; set; }
        public int[][] ErrorJagged { get; set; }
        public double[] TrueValues { get; set; }
        public double[][] TrueValuesJagged { get; set; }
        public double[] ObsValues { get; set; }
        public double[][] ObsValuesJagged { get; set; }

        public static CitizenData FromFile(string fileName)
        {
            string[] data = File.ReadAllLines(fileName);
            int numberOfData = data.Length - 1;
            var citizenIDs = new int[numberOfData];
            var trueValues = new double[numberOfData];
            var obsValues = new double[numberOfData];
            var error = new int[numberOfData];
            var eventIDs = new int[numberOfData];
            for (int i = 0; i < numberOfData; i++)
            {
                string[] record = data[i + 1].Split('\t');//assumes tab-delimited data
                citizenIDs[i] = int.Parse(record[0]);
                eventIDs[i] = int.Parse(record[4]);
                trueValues[i] = double.Parse(record[1]);
                obsValues[i] = double.Parse(record[2]);
                error[i] = int.Parse(record[3]);

            }
            return new CitizenData { CitizenIDs = citizenIDs, TrueIndex = eventIDs, TrueValues = trueValues, ObsValues = obsValues, Error = error }.MakeJagged();
        }

        private CitizenData MakeJagged()
        {
            var startIndex = new List<int> { 0 };
            int oldID = CitizenIDs[0];
            for (int i = 1; i < CitizenIDs.Length; i++)
            {
                int newID = CitizenIDs[i];
                if (newID != oldID) startIndex.Add(i);
                oldID = newID;
            }
            startIndex.Add(CitizenIDs.Length);

            TrueIndexJagged = Utils.CreateArray(startIndex.Count - 1, i => TrueIndex.SubArray(startIndex[i], startIndex[i + 1] - startIndex[i]));
            TrueValuesJagged = Utils.CreateArray(startIndex.Count - 1, i => TrueValues.SubArray(startIndex[i], startIndex[i + 1] - startIndex[i]));
            ObsValuesJagged = Utils.CreateArray(startIndex.Count - 1, i => ObsValues.SubArray(startIndex[i], startIndex[i + 1] - startIndex[i]));
            ErrorJagged = Utils.CreateArray(startIndex.Count - 1, i => Error.SubArray(startIndex[i], startIndex[i + 1] - startIndex[i]));

            return this;
            }
        }
    }

