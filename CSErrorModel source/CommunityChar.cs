/// This is the Citizen Science Error Model developed by Gerrit Schoups and Jessica Eisma with valuable input from
/// Nick van de Geisen and Jeff Davids. Please cite our work if you use it. A description of the model can be found in:
/// 
/// Eisma, J. A., Schoups, G., Davids, J. C., and van de Giesen, N.: A Bayesian model for quantifying errors in citizen science 
/// data: Application to rainfall observations from Nepal, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-658, 2023. 
/// 
/// Here, first the model is developed, then the method for loading and organizing the input data is provided, then the model is run.
/// Lastly, model results are printed to the console and to a text file.
/// 
/// To set up Visual Studio to use Infer.NET, see https://github.com/dotnet/infer. Infer.NET is not our work. Please cite Infer.NET as
/// indicated here: https://dotnet.github.io/infer/userguide/Frequently%20Asked%20Questions.html

using Microsoft.ML.Probabilistic.Models;

namespace CSErrorModel
{
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Compiler.Visualizers;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Range = Range;

    public class CommunityCharModel
    {
        /// <summary>
        /// Defining model variables
        /// </summary>
        public VariableArray<int> Community;    // Error communities      
        public VariableArray<VariableArray<int>, int[][]> Error;    // No Error(0), Unit Error(1), Meniscus Error(2), Unknown Error(3)
        public VariableArray<int> CitizenSciID;                     // Unique ID for citizen scientists
        public VariableArray<double> TrueValue;                     // Rainfall value from the submitted photo
        public VariableArray<double> aSlope;                        // Slope of the regression equation
        public VariableArray<double> bInt;                          // Intercept of the regression equation
        public VariableArray<double> Precs;                         // Precision of the gaussian for the regression
        public VariableArray<double> alphaPrior;                    // Shape parameter for the gamma prior for the Precs variable
        public VariableArray<double> betaPrior;                     // Scale parameter for the gamma prior for the Precs variable  
        public Variable<int> CSISizeVar;                            // Number of CS; outer index for jaggedCSObs array   
        public VariableArray<int> CSObsSizesVar;                    // Number of observations that each CS has submitted; inner index for jaggedCSObs array
        public VariableArray<VariableArray<double>, double[][]> jaggedCSObs;    // Value of submitted observation
        public VariableArray<VariableArray<int>, int[][]> CSEindex;             // Indexes each event

        /// <summary>
        /// Defining CS characteristics
        /// </summary>
        public VariableArray<int> Motivation;   // Paid, Volunteer
        public VariableArray<int> Recruitment;  // Outreach, Personal connection, Random visit, Social media 
        public VariableArray<int> Age;          // <=18, 19-25, >25
        public VariableArray<int> Education;    // <Bachelors, Bachelors, >Bachelors
        public VariableArray<int> Occupation;   // Student, Agriculture, Other
        public VariableArray<int> RuralUrban;   // Rural, Urban, Semi-Urban
        public VariableArray<int> Gender;       // Male, Female
        public VariableArray<int> Ability;      // Performance ratio from Davids et al. (2019)
        public VariableArray<int> Experience;   // Number of submissions

        /// <summary>
        /// Random variables representing the parameters of the distributions of the primary random variables. 
        /// For child variables, these are in the form of conditional probability tables (CPTs)
        /// </summary>
        public VariableArray<Discrete> ProbCommunity;
        public VariableArray<Vector> CPTError;
        public VariableArray<Vector> CPTMotivation;
        public VariableArray<Vector> CPTRecruitment;
        public VariableArray<Vector> CPTAge;
        public VariableArray<Vector> CPTEducation;
        public VariableArray<Vector> CPTOccupation;
        public VariableArray<Vector> CPTRuralUrban;
        public VariableArray<Vector> CPTGender;
        public VariableArray<Vector> CPTAbility;
        public VariableArray<Vector> CPTExperience;

        /// <summary>
        /// Defining Prior distributions
        /// </summary>
        public VariableArray<Dirichlet> CPTErrorPrior;
        public VariableArray<Gaussian> TrueValuePrior;
        public VariableArray<VariableArray<Gaussian>, Gaussian[][]> CSObservationPrior;
        public VariableArray<Dirichlet> CPTMotivationPrior;
        public VariableArray<Dirichlet> CPTRecruitmentPrior;
        public VariableArray<Dirichlet> CPTAgePrior;
        public VariableArray<Dirichlet> CPTEducationPrior;
        public VariableArray<Dirichlet> CPTOccupationPrior;
        public VariableArray<Dirichlet> CPTRuralUrbanPrior;
        public VariableArray<Dirichlet> CPTGenderPrior;
        public VariableArray<Dirichlet> CPTAbilityPrior;
        public VariableArray<Dirichlet> CPTExperiencePrior;
        public VariableArray<Gaussian> aSlopePrior;
        public VariableArray<Gaussian> bIntPrior;
        public VariableArray<Gamma> PrecsPrior;

        /// <summary>
        /// Defining Posterior distributions
        /// </summary>
        public Discrete[] CommunityPosterior;
        public Dirichlet[] CPTErrorPosterior;
        public Gaussian[] TrueValuePosterior;
        public Gaussian[][] CSobservationPosterior;
        public Dirichlet[] CPTMotivationPosterior;
        public Dirichlet[] CPTRecruitmentPosterior;
        public Dirichlet[] CPTAgePosterior;
        public Dirichlet[] CPTEducationPosterior;
        public Dirichlet[] CPTOccupationPosterior;
        public Dirichlet[] CPTRuralUrbanPosterior;
        public Dirichlet[] CPTGenderPosterior;
        public Dirichlet[] CPTAbilityPosterior;
        public Dirichlet[] CPTExperiencePosterior;
        public Gaussian[] aSlopePosterior;
        public Gaussian[] bIntPosterior;
        public Gamma[] PrecsPosterior;
        public Discrete[][] ErrorPosterior;

        /// <summary>
        /// For calculating model evidence, used when selecting the number of errors and number of communities
        /// </summary>
        public Variable<bool> evidence;
        public IfBlock block;

        public Variable<IDistribution<int[]>> CSCommunityInitializer { get; set; }
        public Range CSI { get; set; }    // Citizen Scientist IDs
        public Range Er { get; set; }     // Error range
        public Range CSObs { get; set; }  // Range of submitted Citizen Scientist Observations
        public Range C { get; set; }      // Range of communities
        public Range Event { get; set; }   // Range of events
        public Range CSE { get; set; }     // Range of indicator variable

        // Inference engine
        public InferenceEngine Engine = new InferenceEngine(new ExpectationPropagation());

        /// <summary>
        /// Constructs a Community/Characteristic model
        /// </summary>
        public void CreateCommunityCharModel(int numCommunity,
        int numMotivation,
        int numRecruitment,
        int numAge,
        int numEducation,
        int numOccupation,
        int numRuralUrban,
        int numGender,
        int numAbility,
        int numExperience,
        int numError,
        int numCitizens,
        int numEvents)
        {
            evidence = Variable.Bernoulli(0.5).Named("evidence");
            block = Variable.If(evidence);

            // Setting up ranges
            Event = new Range(numEvents).Named("Event");       // Event Count
            Er = new Range(numError).Named("Er");                   // Range of Error types
            C = new Range(numCommunity).Named("C");                 // Communities
            Range M = new Range(numMotivation).Named("M");          // Motivation
            Range R = new Range(numRecruitment).Named("R");         // Recruitment
            Range A = new Range(numAge).Named("A");                 // Age
            Range E = new Range(numEducation).Named("E");           // Education
            Range O = new Range(numOccupation).Named("O");          // Occupation
            Range Ru = new Range(numRuralUrban).Named("Ru");        // RuralUrban
            Range G = new Range(numGender).Named("G");              // Gender
            Range Ab = new Range(numAbility).Named("Ab");           // Ability
            Range Ex = new Range(numExperience).Named("Ex");        // Experience

            // Setting up the ranges for a jagged array (for Error variable)
            CSISizeVar = Variable.New<int>().Named("CSISizeVar");
            CSI = new Range(CSISizeVar).Named("CSI");
            CSObsSizesVar = Variable.Array<int>(CSI).Named("CSObsSizesVar");
            CSObs = new Range(CSObsSizesVar[CSI]).Named("CSObs");

            jaggedCSObs = Variable.Array(Variable.Array<double>(CSObs), CSI).Named("jaggedCSObs");

            CSEindex = Variable.Array(Variable.Array<int>(CSObs), CSI).Named("CSEindex");
            CSEindex.SetValueRange(Event);

            ProbCommunity = Variable.Array<Discrete>(CSI).Named("ProbCommunity");
            Community = Variable.Array<int>(CSI).Named("Community");
            Community[CSI] = Variable<int>.Random(ProbCommunity[CSI]); // ProbCommunity
            Community.SetValueRange(C);

            // Symmetry breaking
            CSCommunityInitializer = Variable.New<IDistribution<int[]>>().Named("CSCommunityInitializer");
            Community.InitialiseTo(CSCommunityInitializer);

            // True Value probability
            TrueValuePrior = Variable.Array<Gaussian>(Event).Named("TrueValuePrior");
            TrueValue = Variable.Array<double>(Event).Named("TrueValue");
            TrueValue[Event] = Variable<double>.Random(TrueValuePrior[Event]);

            /// <summary>
            /// Set constraints on the inferred true value specific to your citizen science program. Here, two constraints are set.
            /// </summary>
            Variable.ConstrainTrue(TrueValue[Event] >= 0);
            Variable.ConstrainBetween(TrueValue[Event], 0, 200); // 200 mm is the maximum measurement possible with the S4W-Nepal rain gauges

            // CSObs probability
            CSObservationPrior = Variable.Array(Variable.Array<Gaussian>(CSObs), CSI).Named("CSObservationPrior");
            jaggedCSObs = Variable.Array(Variable.Array<double>(CSObs), CSI).Named("jaggedCSObs");

            // Error probability table conditioned on community and SubmittedObservation
            CPTErrorPrior = Variable.Array<Dirichlet>(C).Named("CPTErrorPrior");
            CPTError = Variable.Array<Vector>(C).Named("CPTError");
            CPTError[C] = Variable<Vector>.Random(CPTErrorPrior[C]);
            CPTError.SetValueRange(Er);

            Error = Variable.Array(Variable.Array<int>(CSObs), CSI).Named("Error");
            Error.SetValueRange(Er);

            // Motivation probability table conditioned on community
            CPTMotivationPrior = Variable.Array<Dirichlet>(C).Named("CPTMotivationPrior");
            CPTMotivation = Variable.Array<Vector>(C).Named("CPTMotivation");
            CPTMotivation[C] = Variable<Vector>.Random(CPTMotivationPrior[C]);
            CPTMotivation.SetValueRange(M);

            // Recruitment probability table conditioned on community
            CPTRecruitmentPrior = Variable.Array<Dirichlet>(C).Named("CPTRecruitmentPrior");
            CPTRecruitment = Variable.Array<Vector>(C).Named("CPTRecruitment");
            CPTRecruitment[C] = Variable<Vector>.Random(CPTRecruitmentPrior[C]);
            CPTRecruitment.SetValueRange(R);

            // Age probability table conditioned on community
            CPTAgePrior = Variable.Array<Dirichlet>(C).Named("CPTAgePrior");
            CPTAge = Variable.Array<Vector>(C).Named("CPTAge");
            CPTAge[C] = Variable<Vector>.Random(CPTAgePrior[C]);
            CPTAge.SetValueRange(A);

            // Education probability table conditioned on community
            CPTEducationPrior = Variable.Array<Dirichlet>(C).Named("CPTEducationPrior");
            CPTEducation = Variable.Array<Vector>(C).Named("CPTEducation");
            CPTEducation[C] = Variable<Vector>.Random(CPTEducationPrior[C]);
            CPTEducation.SetValueRange(E);

            // Occupation probability table conditioned on community
            CPTOccupationPrior = Variable.Array<Dirichlet>(C).Named("CPTOccupationPrior");
            CPTOccupation = Variable.Array<Vector>(C).Named("CPTOccupation");
            CPTOccupation[C] = Variable<Vector>.Random(CPTOccupationPrior[C]);
            CPTOccupation.SetValueRange(O);

            // RuralUrban probability table conditioned on community
            CPTRuralUrbanPrior = Variable.Array<Dirichlet>(C).Named("CPTRuralUrbanPrior");
            CPTRuralUrban = Variable.Array<Vector>(C).Named("CPTRuralUrban");
            CPTRuralUrban[C] = Variable<Vector>.Random(CPTRuralUrbanPrior[C]);
            CPTRuralUrban.SetValueRange(Ru);

            // Gender probability table conditioned on community
            CPTGenderPrior = Variable.Array<Dirichlet>(C).Named("CPTGenderPrior");
            CPTGender = Variable.Array<Vector>(C).Named("CPTGender");
            CPTGender[C] = Variable<Vector>.Random(CPTGenderPrior[C]);
            CPTGender.SetValueRange(G);

            // Ability probability table conditioned on community
            CPTAbilityPrior = Variable.Array<Dirichlet>(C).Named("CPTAbilityPrior");
            CPTAbility = Variable.Array<Vector>(C).Named("CPTAbility");
            CPTAbility[C] = Variable<Vector>.Random(CPTAbilityPrior[C]);
            CPTAbility.SetValueRange(Ab);

            // Experience probability table conditioned on community
            CPTExperiencePrior = Variable.Array<Dirichlet>(C).Named("CPTExperiencePrior");
            CPTExperience = Variable.Array<Vector>(C).Named("CPTExperience");
            CPTExperience[C] = Variable<Vector>.Random(CPTExperiencePrior[C]);
            CPTExperience.SetValueRange(Ex);

            // Define the structure/primary variables
            Motivation = AddChildFromOneParent(Community, CPTMotivation).Named("Motivation");
            Recruitment = AddChildFromOneParent(Community, CPTRecruitment).Named("Recruitment");
            Age = AddChildFromOneParent(Community, CPTAge).Named("Age");
            Education = AddChildFromOneParent(Community, CPTEducation).Named("Education");
            Occupation = AddChildFromOneParent(Community, CPTOccupation).Named("Occupation");
            RuralUrban = AddChildFromOneParent(Community, CPTRuralUrban).Named("RuralUrban");
            Gender = AddChildFromOneParent(Community, CPTGender).Named("Gender");
            Ability = AddChildFromOneParent(Community, CPTAbility).Named("Ability");
            Experience = AddChildFromOneParent(Community, CPTExperience).Named("Experience");

            /// <Summary>
            /// Make the switch around the Error variable
            /// </Summary>
            using (Variable.ForEach(CSI))
            {
                using (Variable.Switch(Community[CSI]))
                {
                    using (Variable.ForEach(CSObs))
                    {
                        Error[CSI][CSObs] = Variable.Discrete(CPTError[Community[CSI]]);
                    }
                }
            }

            /// <Summary>
            /// Make the switch around the Regression factor node
            /// Number of mixture components = Er
            /// </Summary>
            // Mixture components a,b
            aSlope = Variable.Array<double>(Er).Named("aSlope");
            aSlopePrior = Variable.Array<Gaussian>(Er).Named("aSlopePrior");
            aSlope[Er] = Variable<double>.Random(aSlopePrior[Er]);

            bInt = Variable.Array<double>(Er).Named("bInt");
            bIntPrior = Variable.Array<Gaussian>(Er).Named("bIntPrior");
            bInt[Er] = Variable<double>.Random(bIntPrior[Er]);

            // Mixture component precisions
            Precs = Variable.Array<double>(Er).Named("Precs");
            PrecsPrior = Variable.Array<Gamma>(Er).Named("PrecsPrior");
            Precs[Er] = Variable<double>.Random(PrecsPrior[Er]);

            /// <Summary>
            /// Mixture weights are defined by the Error posterior. Mixture of Gaussians model
            /// </Summary>
            using (Variable.ForEach(CSI))
            {
                using (Variable.ForEach(CSObs))
                {
                    using (Variable.Switch(Error[CSI][CSObs]))
                    {
                        jaggedCSObs[CSI][CSObs] = Variable.GaussianFromMeanAndPrecision(aSlope[Error[CSI][CSObs]] * TrueValue[CSEindex[CSI][CSObs]] + bInt[Error[CSI][CSObs]], Precs[Error[CSI][CSObs]]);
                    }
                }
            }
            block.CloseBlock();
        }

        /// <summary>
        /// Learns the parameters of CommunityCharModel
        /// </summary>
        public void LearnParameters(
        string RunType,     // "train" or "test" --> write these in lines 677 and 721. Indicates whether this is the training or testing run of the model.
        int numCS_test,
        int[] CitizenSciIDChar,
        int[] motivation,
        int[] recruitment,
        int[] age,
        int[] education,
        int[] occupation,
        int[] ruralurban,
        int[] gender,
        int[] ability,
        int[] experience,
        int[][] error,
        double[][] jaggedCSO,
        double[] csObservation,
        double[] csObservation_test,
        double[] alphaprior, // start of variables for training data
        double[] betaprior,
        double meantvprior,
        double vartvprior,
        double[] meanaprior,
        double[] varaprior,
        double[] meanbprior,
        double[] varbprior,
        double[] Truevalue, // end of variables for training data
        double[] Truevalue_test,
        int[][] cseIndex,
        int[] CitizenSciID_testChar,
        Discrete[] communityPosterior,
        Dirichlet[] cptMotivationPosterior,
        Dirichlet[] cptRecruitmentPosterior,
        Dirichlet[] cptAgePosterior,
        Dirichlet[] cptEducationPosterior,
        Dirichlet[] cptOccupationPosterior,
        Dirichlet[] cptRuralUrbanPosterior,
        Dirichlet[] cptGenderPosterior,
        Dirichlet[] cptAbilityPosterior,
        Dirichlet[] cptExperiencePosterior,
        Dirichlet[] cptErrorPosterior,
        Gaussian[] trueValuePosterior,
        Gaussian[] aSlopePosterior_,
        Gaussian[] bIntPosterior_,
        Gamma[] PrecsPosterior_)   // end for testing data

        {

            CSEindex.ObservedValue = cseIndex;

            // Set the uniform priors for the probability tables
            int numCommunity = Community.GetValueRange().SizeAsInt;
            int numMotivation = Motivation.GetValueRange().SizeAsInt;
            int numRecruitment = Recruitment.GetValueRange().SizeAsInt;
            int numAge = Age.GetValueRange().SizeAsInt;
            int numEducation = Education.GetValueRange().SizeAsInt;
            int numOccupation = Occupation.GetValueRange().SizeAsInt;
            int numRuralUrban = RuralUrban.GetValueRange().SizeAsInt;
            int numGender = Gender.GetValueRange().SizeAsInt;
            int numAbility = Ability.GetValueRange().SizeAsInt;
            int numExperience = Experience.GetValueRange().SizeAsInt;
            int numError = Error.GetValueRange().SizeAsInt;
            int numEvents = CSEindex.GetValueRange().SizeAsInt;
            int numCS = recruitment.Length;
            int numTV_train = Truevalue.Length; // for estimating truevalueprior for test
            int numCSO_train = csObservation.Length;

            if (RunType == "train")
            {
                // Set Observations for training
                TrueValue.ObservedValue = Truevalue;
                Motivation.ObservedValue = motivation;
                Recruitment.ObservedValue = recruitment;
                Age.ObservedValue = age;
                Education.ObservedValue = education;
                Occupation.ObservedValue = occupation;
                RuralUrban.ObservedValue = ruralurban;
                Gender.ObservedValue = gender;
                Ability.ObservedValue = ability;
                Experience.ObservedValue = experience;

                // Create variables to hold uniform priors
                Discrete[] probCommunity = new Discrete[numCS];
                Dirichlet[] cptMotivationPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptRecruitmentPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptAgePrior = new Dirichlet[numCommunity];
                Dirichlet[] cptEducationPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptOccupationPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptRuralUrbanPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptGenderPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptAbilityPrior = new Dirichlet[numCommunity];
                Dirichlet[] cptExperiencePrior = new Dirichlet[numCommunity];
                Dirichlet[] cptErrorPrior = new Dirichlet[numCommunity];
                Gaussian[] Truevalueprior = new Gaussian[numEvents];
                Gaussian[] aSlopeprior = new Gaussian[numError];
                Gaussian[] bIntprior = new Gaussian[numError];
                Gamma[] precsPrior = new Gamma[numError];

                // Assign uniform priors to variables
                for (int i = 0; i < numCS; i++)
                {
                    probCommunity[i] = Discrete.Uniform(numCommunity);
                }

                for (int i = 0; i < numCommunity; i++)
                {
                    cptMotivationPrior[i] = Dirichlet.Uniform(numMotivation);
                    cptRecruitmentPrior[i] = Dirichlet.Uniform(numRecruitment);
                    cptAgePrior[i] = Dirichlet.Uniform(numAge);
                    cptEducationPrior[i] = Dirichlet.Uniform(numEducation);
                    cptOccupationPrior[i] = Dirichlet.Uniform(numOccupation);
                    cptRuralUrbanPrior[i] = Dirichlet.Uniform(numRuralUrban);
                    cptGenderPrior[i] = Dirichlet.Uniform(numGender);
                    cptAbilityPrior[i] = Dirichlet.Uniform(numAbility);
                    cptExperiencePrior[i] = Dirichlet.Uniform(numExperience);
                    cptErrorPrior[i] = Dirichlet.Uniform(numError);
                }
                for (int j = 0; j < numEvents; j++)
                {
                    Truevalueprior[j] = Gaussian.FromMeanAndPrecision(meantvprior, 1 / vartvprior);
                }

                for (int t = 0; t < (numError); t++)
                {
                    aSlopeprior[t] = Gaussian.FromMeanAndPrecision(meanaprior[t], 1 / varaprior[t]);
                    bIntprior[t] = Gaussian.FromMeanAndPrecision(meanbprior[t], 1 / varbprior[t]);
                    precsPrior[t] = Gamma.FromShapeAndRate(alphaprior[t], betaprior[t]);
                }

                Console.WriteLine("Observing Training Parameters");

                CSISizeVar.ObservedValue = jaggedCSO.Length;
                var CSObsSizes1 = new int[jaggedCSO.Length];
                for (int i = 0; i < jaggedCSO.Length; i++)
                    CSObsSizes1[i] = jaggedCSO[i].Length;
                CSObsSizesVar.ObservedValue = CSObsSizes1;
                jaggedCSObs.ObservedValue = jaggedCSO;

                // Observe the uniform priors
                ProbCommunity.ObservedValue = probCommunity;
                CPTMotivationPrior.ObservedValue = cptMotivationPrior;
                CPTRecruitmentPrior.ObservedValue = cptRecruitmentPrior;
                CPTAgePrior.ObservedValue = cptAgePrior;
                CPTEducationPrior.ObservedValue = cptEducationPrior;
                CPTOccupationPrior.ObservedValue = cptOccupationPrior;
                CPTRuralUrbanPrior.ObservedValue = cptRuralUrbanPrior;
                CPTGenderPrior.ObservedValue = cptGenderPrior;
                CPTAbilityPrior.ObservedValue = cptAbilityPrior;
                CPTExperiencePrior.ObservedValue = cptExperiencePrior;
                CPTErrorPrior.ObservedValue = cptErrorPrior;
                TrueValuePrior.ObservedValue = Truevalueprior;
                aSlopePrior.ObservedValue = aSlopeprior;
                bIntPrior.ObservedValue = bIntprior;
                PrecsPrior.ObservedValue = precsPrior;
            }
            else
            {
                Console.WriteLine("Observing Test Parameters");
                ProbCommunity.ObservedValue = Util.ArrayInit(CitizenSciID_testChar.Length, i => communityPosterior[Array.IndexOf(CitizenSciIDChar, CitizenSciID_testChar[i])]);

                CPTMotivationPrior.ObservedValue = cptMotivationPosterior;
                CPTRecruitmentPrior.ObservedValue = cptRecruitmentPosterior;
                CPTAgePrior.ObservedValue = cptAgePosterior;
                CPTEducationPrior.ObservedValue = cptEducationPosterior;
                CPTOccupationPrior.ObservedValue = cptOccupationPosterior;
                CPTRuralUrbanPrior.ObservedValue = cptRuralUrbanPosterior;
                CPTGenderPrior.ObservedValue = cptGenderPosterior;
                CPTAbilityPrior.ObservedValue = cptAbilityPosterior;
                CPTExperiencePrior.ObservedValue = cptExperiencePosterior;
                CPTErrorPrior.ObservedValue = cptErrorPosterior;
                aSlopePrior.ObservedValue = Util.ArrayInit(aSlopePosterior_.Length, i => Gaussian.PointMass(aSlopePosterior_[i].GetMean()));
                bIntPrior.ObservedValue = bIntPosterior_;
                PrecsPrior.ObservedValue = PrecsPosterior_;

                Gaussian[] Truevalueprior = new Gaussian[numEvents];
                GaussianEstimator est = new GaussianEstimator();
                for (int i = 0; i < numTV_train; i++)
                {
                    double stv = Truevalue[i];
                    est.Add(stv);
                }

                for (int j = 0; j < numEvents; j++)
                {
                    Truevalueprior[j] = Gaussian.FromMeanAndPrecision(meantvprior, 1 / vartvprior);
                }
                TrueValuePrior.ObservedValue = Truevalueprior;
                Truevalue = Truevalue_test;
            }

            /// The observed values can then be set before inference, making sure that the jagged sizes, and the jagged array itself are consistently set:
            /// Set in the learn parameters phase.
            CSISizeVar.ObservedValue = jaggedCSO.Length;
            var CSObsSizes = new int[jaggedCSO.Length];
            for (int i = 0; i < jaggedCSO.Length; i++)
                CSObsSizes[i] = jaggedCSO[i].Length;
            CSObsSizesVar.ObservedValue = CSObsSizes;
            jaggedCSObs.ObservedValue = jaggedCSO;

            // Initialize messages
            Rand.Restart(12347);
            var discreteUniform = Discrete.Uniform(numCommunity);
            CSCommunityInitializer.ObservedValue = Distribution<int>.Array(Util.ArrayInit(jaggedCSO.Length, w => Discrete.PointMass(discreteUniform.Sample(), numCommunity)));

            // Inference Settings
            Engine.ShowProgress = true;
            Engine.ShowTimings = false;
            Engine.ShowMsl = false;
            Engine.ShowFactorGraph = false; //Turning this on (true) slows down the program significantly.
            Engine.ShowSchedule = false;
            Engine.SaveFactorGraphToFolder = "graphs";
            Engine.NumberOfIterations = 50;

            /// Compute model evidence. Used to determine the number of parameters
            if (RunType == "train")
            {
                double logEvidence = Engine.Infer<Bernoulli>(evidence).LogOdds;
                Console.WriteLine("Evidence: " + logEvidence);
            }

            /// Running the inference engine
            Console.WriteLine("Performing Inference " + RunType);
            CommunityPosterior = Engine.Infer<Discrete[]>(Community);
            Console.WriteLine("Inferring TV");
            TrueValuePosterior = Engine.Infer<Gaussian[]>(TrueValue);
            CPTMotivationPosterior = Engine.Infer<Dirichlet[]>(CPTMotivation);
            CPTRecruitmentPosterior = Engine.Infer<Dirichlet[]>(CPTRecruitment);
            CPTAgePosterior = Engine.Infer<Dirichlet[]>(CPTAge);
            CPTEducationPosterior = Engine.Infer<Dirichlet[]>(CPTEducation);
            CPTOccupationPosterior = Engine.Infer<Dirichlet[]>(CPTOccupation);
            CPTRuralUrbanPosterior = Engine.Infer<Dirichlet[]>(CPTRuralUrban);
            CPTGenderPosterior = Engine.Infer<Dirichlet[]>(CPTGender);
            CPTAbilityPosterior = Engine.Infer<Dirichlet[]>(CPTAbility);
            CPTExperiencePosterior = Engine.Infer<Dirichlet[]>(CPTExperience);
            CPTErrorPosterior = Engine.Infer<Dirichlet[]>(CPTError);
            aSlopePosterior = Engine.Infer<Gaussian[]>(aSlope);
            bIntPosterior = Engine.Infer<Gaussian[]>(bInt);
            PrecsPosterior = Engine.Infer<Gamma[]>(Precs);
            ErrorPosterior = Engine.Infer<Discrete[][]>(Error);
        }

        /// <summary>
        /// Helper method to add a child from one parent
        /// </summary>
        /// <param name="parent">Parent (a variable array over a range of examples)</param>
        /// <param name="cpt">Conditional probability table</param>
        /// <returns></returns>
        public static VariableArray<int> AddChildFromOneParent(
            VariableArray<int> parent,
            VariableArray<Vector> cpt)
        {
            var n = parent.Range;
            var child = Variable.Array<int>(n);
            using (Variable.ForEach(n))
            using (Variable.Switch(parent[n]))
            {
                child[n] = Variable.Discrete(cpt[parent[n]]);
            }
            return child;
        }
    }

    public class CommunityChar
    {
        public static void Main(string[] args)
        {
            ///
            /// Replace the values here as appropriate. For example, how many communities are there? How many categories do you have for
            /// Motivation? Recruitment? etc.
            /// 
            /// For the priors, put whatever is reasonable for your dataset. The priors are simply a starting point; the model will
            /// infer the correct value. However, it is important to note that the priors may affect the results and will likely influence 
            /// how many iterations are required for the model to converge. Run model multiple times to be sure that the priors are not 
            /// significantly altering the results. The goal is a stable model result.
            ///
            int numCommunity = 4;
            int numMotivation = 2;
            int numRecruitment = 4;
            int numAge = 3;
            int numEducation = 3;
            int numOccupation = 3;
            int numRuralUrban = 3;
            int numGender = 2;
            int numAbility = 3;
            int numExperience = 3;
            int numError = 5;
            int numCitizens = 152; 
            int numEvents = 6091; 
            double[] alphaprior = { 0.25, 0.75, 1.5, 0.5 , 15 };
            double[] betaprior = { 0.05, 0.25, 0.05 , 0.01, 10 }; 
            double meantvprior = 15;
            double vartvprior = 2400;
            double[] meanaprior = { 1, 0.1, 1.002, 0.9, 7 }; // for the regression mix
            double[] varaprior = { 0.5, 0.5, 2.0, 50, 70 }; 
            double[] meanbprior = { 0, 0.02, 2.3, 4.2, 3 }; 
            double[] varbprior = { 0.5, 0.5, 0.2, 50, 30 };

            InferenceEngine.Visualizer = new WindowsVisualizer();

            /// Create a new model
            CommunityCharModel CommChar = new CommunityCharModel();
            CommChar.CreateCommunityCharModel(numCommunity, numMotivation, numRecruitment, numAge, numEducation, numOccupation, numRuralUrban, numGender, numAbility, numExperience, numError, numCitizens, numEvents);

            ///
            /// Load the Training Data. Need CS observations and true value in one file. Need CS characteristics in another file.
            /// 

            var citizenCharacteristics = CitizenCharacteristics.FromFile($"../../../bin/Debug/net461/Data/Group3TrainChar.txt");

            int[] CitizenSciIDChar = citizenCharacteristics.CitizenSciID;
            int[] motivation = citizenCharacteristics.motivation;
            int[] recruitment = citizenCharacteristics.recruitment;
            int[] age = citizenCharacteristics.age;
            int[] education = citizenCharacteristics.education;
            int[] occupation = citizenCharacteristics.occupation;
            int[] ruralurban = citizenCharacteristics.ruralurban;
            int[] gender = citizenCharacteristics.gender;
            int[] ability = citizenCharacteristics.ability;
            int[] experience = citizenCharacteristics.experience;

            var citizenData = CitizenData.FromFile($"../../../bin/Debug/net461/Data/Group3Train.txt");

            int[] CitizenSciID = citizenData.CitizenIDs;
            double[] csObservation = citizenData.ObsValues;
            double[] Truevalue = citizenData.TrueValues;
            double[][] jaggedCSO = citizenData.ObsValuesJagged;
            double[][] jaggedTV = citizenData.TrueValuesJagged;
            int[][] error = citizenData.ErrorJagged;
            int[][] cseIndex = citizenData.TrueIndexJagged;
     
            // Uniform dist. placeholders
            Dirichlet DirPriorVal = Dirichlet.Uniform(numCommunity);
            Dirichlet[] DirPrior = new Dirichlet[numCommunity];
            Discrete[] DiscPrior = new Discrete[numCommunity];
            Gaussian[] GauPrior = new Gaussian[numEvents];
            Gamma[] GamPrior = new Gamma[numError];

            for (int i = 0; i < numCommunity; i++)
            {
                DirPrior[i] = Dirichlet.Uniform(numMotivation);

            }
            for (int j = 0; j < numEvents; j++)
            {
                GauPrior[j] = Gaussian.FromMeanAndPrecision(0, 1);
            }
            for (int t = 0; t < (numError); t++)
            {
                GamPrior[t] = Gamma.FromShapeAndRate(1, 0.5);
            }

            CommChar.LearnParameters("train", numCitizens, CitizenSciIDChar, motivation, recruitment, age, education, occupation, ruralurban, gender, ability, experience, error, jaggedCSO, csObservation, csObservation, alphaprior, betaprior, meantvprior, vartvprior, meanaprior, varaprior, meanbprior, varbprior, Truevalue, Truevalue, cseIndex, CitizenSciIDChar,
                DiscPrior, DirPrior, DirPrior, DirPrior, DirPrior, DirPrior, DirPrior, DirPrior, DirPrior, DirPrior, DirPrior, GauPrior, GauPrior, GauPrior, GamPrior); 
            
            /////////////////////////////////////////////////////
            ///////////////////Testing Data//////////////////////
            /////////////////////////////////////////////////////

            ///
            /// These are the only values that should change, based on the amount of data in the testing dataset.
            ///
            int numEvents_test = 527;  
            int numCitizens_test = 109;     

            CommunityCharModel CommCharTest = new CommunityCharModel();
            CommCharTest.CreateCommunityCharModel(numCommunity, numMotivation, numRecruitment, numAge, numEducation, numOccupation, numRuralUrban, numGender, numAbility, numExperience, numError, numCitizens_test, numEvents_test); 

            ///
            /// Load the Testing Data. Need CS observations and true value in one file. Need CS characteristics in another file. 
            /// 

            var citizenCharacteristics_test = CitizenCharacteristics.FromFile($"../../../bin/Debug/net461/Data/Group3TestChar.txt");

            int[] CitizenSciID_testChar = citizenCharacteristics_test.CitizenSciID;
            int[] motivation_test = citizenCharacteristics_test.motivation;
            int[] recruitment_test = citizenCharacteristics_test.recruitment;
            int[] age_test = citizenCharacteristics_test.age;
            int[] education_test = citizenCharacteristics_test.education;
            int[] occupation_test = citizenCharacteristics_test.occupation;
            int[] ruralurban_test = citizenCharacteristics_test.ruralurban;
            int[] gender_test = citizenCharacteristics_test.gender;
            int[] ability_test = citizenCharacteristics_test.ability;
            int[] experience_test = citizenCharacteristics_test.experience;

            var citizenData_test = CitizenData.FromFile($"../../../bin/Debug/net461/Data/Group3Test.txt");

            int[] CitizenSciID_test = citizenData_test.CitizenIDs;
            double[] csObservation_test = citizenData_test.ObsValues;
            double[] Truevalue_test = citizenData_test.TrueValues;
            double[][] jaggedCSO_test = citizenData_test.ObsValuesJagged;
            double[][] jaggedTV_test = citizenData_test.TrueValuesJagged;
            int[][] error_test = citizenData_test.ErrorJagged;
            int[][] cseIndex_test = citizenData_test.TrueIndexJagged;

            /// Change the characteristics variables to the training data
            CommCharTest.LearnParameters("test", 
                numCitizens_test, 
                CitizenSciIDChar, 
                motivation_test, 
                recruitment_test, 
                age_test, 
                education_test, 
                occupation_test, 
                ruralurban_test, 
                gender_test, 
                ability_test, 
                experience_test, 
                error_test, 
                jaggedCSO_test, 
                csObservation, 
                csObservation_test, 
                alphaprior, 
                betaprior, 
                meantvprior, 
                vartvprior, 
                meanaprior, 
                varaprior, 
                meanbprior, 
                varbprior, 
                Truevalue, 
                Truevalue_test, 
                cseIndex_test, 
                CitizenSciID_testChar,
                CommChar.CommunityPosterior,
                CommChar.CPTMotivationPosterior,
                CommChar.CPTRecruitmentPosterior,
                CommChar.CPTAgePosterior,
                CommChar.CPTEducationPosterior,
                CommChar.CPTOccupationPosterior,
                CommChar.CPTRuralUrbanPosterior,
                CommChar.CPTGenderPosterior,
                CommChar.CPTAbilityPosterior,
                CommChar.CPTExperiencePosterior,
                CommChar.CPTErrorPosterior,
                CommChar.TrueValuePosterior,
                CommChar.aSlopePosterior,
                CommChar.bIntPosterior,
                CommChar.PrecsPosterior);

            /////////////////////////////////////////////////////////////////////////////////////////
            //////////////////Printing Results to the Console or Text File///////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////

            Console.WriteLine("\n*********************************************");
            Console.WriteLine("Learning parameters from data (uniform prior)");
            Console.WriteLine("*********************************************");

            var lstCS_Com = new List<double>();
            Console.WriteLine("Community Statistics");
            for (int i = 0; i < numCitizens; i++)
            {
                lstCS_Com.Add(CommChar.CommunityPosterior[i].GetMode());
            }
            var g = lstCS_Com.GroupBy(i => i);

            Console.WriteLine("{0,-10} {1,-10} {2,-10:F2}","Community","CS Count", "Prob.");
            foreach (var grp in g)
            {
                Console.WriteLine("{0,-10} {1,-10} {2,-10:F2}", grp.Key, grp.Count(), (grp.Count()/Convert.ToDouble(numCitizens)).ToString("F2"));
            }
            
            char[] sep = { ',' };
            string errors;
            errors = "0,1,2,3,4";
            string[] mySplitStrEr = errors.Split(sep);
            Console.WriteLine("\nRegression Posteriors");
            Console.WriteLine("{0,-12}{1,-10}{2,-10}{3,-10}", "Error Type", "aSlope", "bInt", "Precision");
            for (int i = 0; i < (numError); i++)
            {
                Console.WriteLine(
                    "{0,-12}{1,-10:F3}{2,-10:F3}{3,-10:F3}",
                    mySplitStrEr[i],
                    CommChar.aSlopePosterior[i].GetMean().ToString("F3"),
                    CommChar.bIntPosterior[i].GetMean().ToString("F3"),
                    CommChar.PrecsPosterior[i].GetMean().ToString("F3"));
            }

            string communities;
            communities ="0,1,2,3";
            string[] mySplitStr = communities.Split(sep);

            Console.WriteLine("{0,-28}{1,-15}{2,-20}{3,-15}{4,-15}{5,-15}", "\nError CPT", "0", "1", "2", "3", "4");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F2}{3,-20:F3}{4,-15:F3}{5,-15:F3}{6,-15:F3}",
                    "Prob. Error | ",
                    mySplitStr[i],
                    CommChar.CPTErrorPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTErrorPosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTErrorPosterior[i].GetMean()[2].ToString("F3"),
                    CommChar.CPTErrorPosterior[i].GetMean()[3].ToString("F3"),
                    CommChar.CPTErrorPosterior[i].GetMean()[4].ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-15}", "\nMotivation CPT", "Volunteer", "Paid");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F3}{3,-15:F3}",
                    "Prob. Mot. | ",
                    mySplitStr[i],
                    CommChar.CPTMotivationPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTMotivationPosterior[i].GetMean()[1].ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-20}{3,-15}{4,-15}", "\nRecruitment CPT", "Outreach", "PersonalConnection", "RandomVisit", "SocialMedia");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F3}{3,-20:F3}{4,-15:F3}{5,-15:F3}",
                    "Prob. Rec. | ",
                    mySplitStr[i],
                    CommChar.CPTRecruitmentPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTRecruitmentPosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTRecruitmentPosterior[i].GetMean()[2].ToString("F3"),
                    CommChar.CPTRecruitmentPosterior[i].GetMean()[3].ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-15}{3,-15}", "\nAge CPT", "<=18", "19-25", ">25");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F3}{3,-15:F3}{4,-15:F3}",
                    "Prob. Age| ",
                    mySplitStr[i],
                    CommChar.CPTAgePosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTAgePosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTAgePosterior[i].GetMean()[2].ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-15}{3,-15}", "\nEducation CPT", "<Bachelors", "Bachelors", ">Bachelors");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F3}{3,-15:F3}{4,-15:F3}",
                    "Prob. Edu.| ",
                    mySplitStr[i],
                    CommChar.CPTEducationPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTEducationPosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTEducationPosterior[i].GetMean()[2].ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-15}{3,-15}", "\nOccupation CPT", "Agriculture", "Student", "Other");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F3}{3,-15:F3}{4,-15:F3}",
                    "Prob. Occ.| ",
                    mySplitStr[i],
                    CommChar.CPTOccupationPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTOccupationPosterior[i].GetMean()[2].ToString("F3"),
                    CommChar.CPTOccupationPosterior[i].GetMean()[1].ToString("F3"));
            }

            Console.WriteLine("{0,-30}{1,-15}{2,-15}{3,-15}", "\nRuralUrban CPT", "Rural", "Semi-Urban", "Urban");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-15}{1,-15}{2,-15:F3}{3,-15:F3}{4,-15:F3}",
                    "Prob. RurUrb| ",
                    mySplitStr[i],
                    CommChar.CPTRuralUrbanPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTRuralUrbanPosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTRuralUrbanPosterior[i].GetMean()[2].ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-15}", "\nGender CPT", "Female", "Male");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F3}{3,-15:F3}",
                    "Prob. Gen.| ",
                    mySplitStr[i],
                    CommChar.CPTGenderPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTGenderPosterior[i].GetMean()[1].ToString("F3"));
            }

            Console.WriteLine("{0,-30}{1,-15}{2,-15}{3,-15}", "\nAbility CPT", "<70%", "70-90%", ">90%");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-15}{1,-15}{2,-15:F3}{3,-15:F3}{4,-15:F3}",
                    "Prob. Abil.| ",
                    mySplitStr[i],
                    CommChar.CPTAbilityPosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTAbilityPosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTAbilityPosterior[i].GetMean()[2].ToString("F3"));
            }

            Console.WriteLine("{0,-52}", "\nObservations");
            Console.WriteLine("{0,-30}{1,-15}{2,-15}{3,-15}", "\nExperience CPT", "<28", "27-53", ">53");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-15}{1,-15}{2,-15:F3}{3,-15:F3}{4,-15:F3}",
                    "Prob. Exp.| ",
                    mySplitStr[i],
                    CommChar.CPTExperiencePosterior[i].GetMean()[0].ToString("F3"),
                    CommChar.CPTExperiencePosterior[i].GetMean()[1].ToString("F3"),
                    CommChar.CPTExperiencePosterior[i].GetMean()[2].ToString("F3"));
            }

            //Writes the enclosed lines to a text file --> too many lines to display in console
            FileStream output1 = new FileStream("ConsoleOutputTrain.txt", FileMode.Create);
            TextWriter outSave = Console.Out;
            StreamWriter portal = new StreamWriter(output1);
            Console.SetOut(portal);

            Console.WriteLine("Inferred Communities");
            for (int i = 0; i < numCitizens; i++)
            {
               Console.WriteLine(
                        "{0,-15}{1,-15}{2,-15}",
                        i,
                        CitizenSciIDChar[i],
                        CommChar.CommunityPosterior[i].GetMode());
            }

            Console.WriteLine("\nInferred Errors");
            for (int i = 0; i < numCitizens; i++)
            {
                for (int j = 0; j < cseIndex[i].Length; j++)
                {
                    Console.WriteLine(
                        "{0,-15}{1,-15}{2,-15}{3,-15}{4,-15}",
                        i, j,
                        CommChar.ErrorPosterior[i][j].GetMode(),
                        jaggedCSO[i][j],
                        jaggedTV[i][j]);
                }
            }
            Console.SetOut(outSave);
            portal.Close();

            Console.WriteLine("\n*********************************************");
            Console.WriteLine("Predicting parameters from Posteriors");
            Console.WriteLine("*********************************************");

            var lstCS_ComTest = new List<double>();
            Console.WriteLine("Community Statistics");
            for (int i = 0; i < numCitizens_test; i++)
            {
                lstCS_ComTest.Add(CommCharTest.CommunityPosterior[i].GetMode());
            }
            var g_test = lstCS_ComTest.GroupBy(i => i);

            Console.WriteLine("{0,-10} {1,-10} {2,-10:F2}", "Community", "CS Count", "Prob.");
            foreach (var grp in g_test)
            {
                Console.WriteLine("{0,-10} {1,-10} {2,-10:F2}", grp.Key, grp.Count(), (grp.Count() / Convert.ToDouble(numCitizens_test)).ToString("F2"));
            }


            Console.WriteLine("\nRegression Posteriors TEST");
            Console.WriteLine("{0,-12}{1,-10}{2,-10}{3,-10}", "Error Type", "aSlope", "bInt", "Precision");
            for (int i = 0; i < (numError); i++)
            {
                Console.WriteLine(
                    "{0,-12}{1,-10:F3}{2,-10:F3}{3,-10:F3}",
                    mySplitStrEr[i],
                    CommCharTest.aSlopePosterior[i].GetMean().ToString("F3"),
                    CommCharTest.bIntPosterior[i].GetMean().ToString("F3"),
                    CommCharTest.PrecsPosterior[i].GetMean().ToString("F3"));
            }

            Console.WriteLine("{0,-28}{1,-15}{2,-20}{3,-15}{4,-15}{5,-15}", "\nError CPT", "0", "1", "2", "3", "4");
            for (int i = 0; i < numCommunity; i++)
            {
                Console.WriteLine(
                    "{0,-13}{1,-15}{2,-15:F2}{3,-20:F3}{4,-15:F3}{5,-15:F3}{6,-15:F3}",
                    "Prob. Error | ",
                    mySplitStr[i],
                    CommCharTest.CPTErrorPosterior[i].GetMean()[0].ToString("F3"),
                    CommCharTest.CPTErrorPosterior[i].GetMean()[1].ToString("F3"),
                    CommCharTest.CPTErrorPosterior[i].GetMean()[2].ToString("F3"),
                    CommCharTest.CPTErrorPosterior[i].GetMean()[3].ToString("F3"),
                    CommCharTest.CPTErrorPosterior[i].GetMean()[4].ToString("F3"));
            }

            /// Writes the enclosed lines to a text file --> too many lines to display in console
            FileStream output1_Test = new FileStream("ConsoleOutputTest.txt", FileMode.Create);
            TextWriter outSave_Test = Console.Out;
            StreamWriter portal_Test = new StreamWriter(output1_Test);
            Console.SetOut(portal_Test);

            Console.WriteLine("Inferred Communities");
            for (int i = 0; i < numCitizens_test; i++)
            {
                Console.WriteLine(
                         "{0,-15}{1,-15}{2,-15}",
                         i,
                         CitizenSciID_testChar[i],
                         CommCharTest.CommunityPosterior[i].GetMode());
            }

            Console.WriteLine("\nInferred Errors");
            for (int i = 0; i < numCitizens_test; i++)
            {
                for (int j = 0; j < cseIndex_test[i].Length; j++)
                {
                    Console.WriteLine(
                        "{0,-15}{1,-15}{2,-15}{3,-15}{4,-15}{5,-15}",
                        i, j,
                        CommCharTest.ErrorPosterior[i][j],
                        CommCharTest.ErrorPosterior[i][j].GetMode(),
                        jaggedCSO_test[i][j],
                        jaggedTV_test[i][j]);
                }
            }

            Console.WriteLine("\nInferred True Values");
            for (int i = 0; i < numEvents_test; i++)
            {
                Console.WriteLine(
                         "{0,-15}{1,-15}{2,-15}{3,-15}{4,-30}",
                         i,
                         CitizenSciID_test[i],
                         CommCharTest.TrueValuePosterior[i],
                         CommCharTest.TrueValuePosterior[i].GetMean(),
                         CommCharTest.TrueValuePosterior[i].GetVariance());
            }
            Console.SetOut(outSave_Test);
            portal.Close();

            Console.WriteLine("\n*********************************************");
            Console.WriteLine("Calculating the Gaussian Mix");
            Console.WriteLine("*********************************************");

            /// For each CS test observation, get a Gaussian mixture posterior distribution for the underlying true value
            double[] a = Util.ArrayInit(numError, i => CommChar.aSlopePosterior[i].GetMean()); // ignore posterior uncertainty of parameters for now
            double[] b = Util.ArrayInit(numError, i => CommChar.bIntPosterior[i].GetMean());
            double[] tau = Util.ArrayInit(numError, i => CommChar.PrecsPosterior[i].GetMean());
            var posteriors = new GaussianMixture[Truevalue_test.Length]; // stores all true value posteriors

            for (int i = 0; i < jaggedCSO_test.Length; i++)
            {
                for (int j = 0; j < jaggedCSO_test[i].Length; j++)
                {
                    double obs = jaggedCSO_test[i][j];// the observation
                    int index = cseIndex_test[i][j];// which true value is this an observation of
                    var weights = CommCharTest.ErrorPosterior[i][j].GetProbs();// posterior error distribution based on test CS characteristics

                    GaussianMixture likelihood = LinearGaussianMixture.Backward(obs, weights, a, b, tau);

                    if (posteriors[index] == null)
                        posteriors[index] = likelihood;
                   else
                        posteriors[index] *= likelihood;// this combines multiple CS likelihoods for the same true value
                }
            }

            // Writes the enclosed lines to a text file --> too many lines to display in console
            FileStream output2_Test = new FileStream("ConsoleOutputTest_GaussianMix.txt", FileMode.Create);
            TextWriter outSave2_Test = Console.Out;
            StreamWriter portal2_Test = new StreamWriter(output2_Test);
            Console.SetOut(portal2_Test);

            // Test: compare posteriors with true values
            for (int i = 0; i < numEvents_test; i++)
            {
                var prior = CommCharTest.TrueValuePrior.ObservedValue[i];  
                posteriors[i].SetToProductWith(prior);// multiply in the prior
                Console.WriteLine($"true value = {Truevalue_test[i]}, posterior = {posteriors[i]}");
            }
            Console.SetOut(outSave2_Test);
            portal.Close();
        }
    }
}




