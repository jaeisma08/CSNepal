using System.IO;

namespace CSErrorModel
{
    public class CitizenCharacteristics
    {
        public int[] CitizenSciID { get; set; }
        public int[] motivation { get; set; }
        public int[] recruitment { get; set; }
        public int[] age { get; set; }
        public int[] education { get; set; }
        public int[] occupation { get; set; }
        public int[] ruralurban { get; set; }
        public int[] gender { get; set; }
        public int[] ability { get; set; }
        public int[] experience { get; set; }

        public static CitizenCharacteristics FromFile(string fileName)
        {
            char[] sep = { '\t', ',' };
            string[] lines = File.ReadAllLines(fileName); // reads entire file into memory
            int numberofCitizens = lines.Length - 1;        // subtract 1 for 
            int[] cs = new int[numberofCitizens];
            int[] mot = new int[numberofCitizens];
            int[] rec = new int[numberofCitizens];
            int[] ag = new int[numberofCitizens];
            int[] edu = new int[numberofCitizens];
            int[] occ = new int[numberofCitizens];
            int[] ru = new int[numberofCitizens];
            int[] gen = new int[numberofCitizens];
            int[] abil = new int[numberofCitizens];
            int[] exp = new int[numberofCitizens];
            for (int i = 0; i < numberofCitizens; i++)
            {
                string[] values = lines[i + 1].Split(sep); // i+1 to skip header
                cs[i] = int.Parse(values[0]);
                mot[i] = int.Parse(values[1]);
                rec[i] = int.Parse(values[2]);
                ag[i] = int.Parse(values[3]);
                edu[i] = int.Parse(values[4]);
                occ[i] = int.Parse(values[5]);
                ru[i] = int.Parse(values[6]);
                gen[i] = int.Parse(values[7]);
                abil[i] = int.Parse(values[8]);
                exp[i] = int.Parse(values[9]);
            }
            return new CitizenCharacteristics { CitizenSciID=cs, motivation=mot, recruitment=rec, age=ag, education=edu, occupation=occ, ruralurban=ru, gender=gen, ability=abil, experience=exp };
        }
    }
}
