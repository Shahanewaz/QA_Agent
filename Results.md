GPT 5.4 Mini:
    Healthcare:
        MedMCQA (validation / dataset): 75.97%
        CAIS and MMLU: 92.64%

    Machine Learning:
        Open Quiz Commons: 99.09%
        CAIS and MMLU: 85.16%

    Cybersecurity:
        Open Quiz Commons: 98.28%
        CAIS and MMLU: 85.34%

    Networking:
        Open Quiz Commons: 98.98%
        PrepBharat: 94%

GPT 4o mini:
    Healthcare:
        MedMCQA (validation / dataset): 67.03%
        CAIS and MMLU: 86.62%

    Machine Learning:
        Open Quiz Commons: 98.18%
        CAIS and MMLU: 70.31%

    Cybersecurity:
        Open Quiz Commons: 96.55%
        CAIS and MMLU: 85.34%

    Networking:
        Open Quiz Commons: 98.98%
        PrepBharat: 92%


GPT 3.5:
    Healthcare:
        MedMCQA (validation / dataset): 55.99%
        CAIS and MMLU: 72.24% 

    Machine Learning:
        Open Quiz Commons: 93.64%
        CAIS and MMLU: 51.56%

    Cybersecurity:
        Open Quiz Commons: 87.93%
        CAIS and MMLU: 75%

    Networking:
        Open Quiz Commons: 92.86%
        PrepBharat: 88%

Open Source Models:

Qwen-3 with 0.6B and 1.7B parameters

Qwen3-0.6B:
    Healthcare:
        MedMCQA (validation / dataset): 25.56%
        CAIS and MMLU: 33.78%

    Machine Learning:
        Open Quiz Commons: 87.27%
        CAIS and MMLU: 30.47%

    Cybersecurity:
        Open Quiz Commons: 63.79%
        CAIS and MMLU: 50.86%

    Networking:
        Open Quiz Commons: 74.49%
        PrepBharat: 50%

Qwen3-1.7B:
    Healthcare:
        MedMCQA (validation / dataset): 23%
        CAIS and MMLU: 40.47%

    Machine Learning:
        Open Quiz Commons: 88.18%
        CAIS and MMLU: 35.16%

    Cybersecurity:
        Open Quiz Commons: 77.59%
        CAIS and MMLU: 51.72%

    Networking:
        Open Quiz Commons: 77.55% 
        PrepBharat: 68%

Gemma2 with 2B and 9B parameters

Gemma2-2B:
    Healthcare:
        MedMCQA (validation / dataset): 41.36%  
        CAIS and MMLU: 47.83%

    Machine Learning:
        Open Quiz Commons: 91.82%
        CAIS and MMLU: 39.06%

    Cybersecurity:
        Open Quiz Commons: 79.31%
        CAIS and MMLU: 56.03%

    Networking:
        Open Quiz Commons: 84.69%
        PrepBharat: 66%

Gemma2-9B:
    Healthcare:
        MedMCQA (validation / dataset): 48.08%  
        CAIS and MMLU: 69.90%

    Machine Learning:
        Open Quiz Commons: 95.45%
        CAIS and MMLU: 56.25%

    Cybersecurity:
        Open Quiz Commons: 87.93%
        CAIS and MMLU: 77.59%

    Networking:
        Open Quiz Commons: 90.82%
        PrepBharat: 80%

Llama2 with 7B and 13B parameters

Llama2-7B:
    Healthcare:
        MedMCQA (validation / dataset): 32.08%
        CAIS and MMLU: 39.46%

    Machine Learning:
        Open Quiz Commons: 69.09%
        CAIS and MMLU: 32.03%

    Cybersecurity:
        Open Quiz Commons: 51.72%
        CAIS and MMLU: 45.69%

    Networking:
        Open Quiz Commons: 58.16%
        PrepBharat: 50%

Llama2-13B:
    Healthcare:
        MedMCQA (validation / dataset):  36%
        CAIS and MMLU: 35.12%

    Machine Learning:
        Open Quiz Commons: 42.73%
        CAIS and MMLU: 29.69%

    Cybersecurity:
        Open Quiz Commons: 34.48%
        CAIS and MMLU: 39.79%

    Networking:
        Open Quiz Commons: 66.33%
        PrepBharat: 44%

Phi3 with 3.8B and 14B parameters 

Phi3-3.8B:
    Healthcare:
        MedMCQA (validation / dataset): 36.72%  
        CAIS and MMLU: 48.16%

    Machine Learning:
        Open Quiz Commons: 61.82%
        CAIS and MMLU: 30.47%

    Cybersecurity:
        Open Quiz Commons: 48.28%
        CAIS and MMLU: 41.38%

    Networking:
        Open Quiz Commons: 64.29%
        PrepBharat: 52%

Phi3-14B:
    Healthcare:
        MedMCQA (validation / dataset): 82.94%
        CAIS and MMLU: 58.09%

    Machine Learning:
        Open Quiz Commons: 94.55%
        CAIS and MMLU: 53.91%

    Cybersecurity:
        Open Quiz Commons: 60.34%
        CAIS and MMLU: 64.66%

    Networking:
        Open Quiz Commons: 83.67%
        PrepBharat: 80%

Phasewise Attack:

Phase 1:

Attack Examples: 5

Phi3-3.8B:
    Healthcare:
        MedMCQA (validation / dataset): 
        CAIS and MMLU: 

    Machine Learning:
        Open Quiz Commons: 
            Baseline C selections : 9 / 110
            Phase1 C selections  : 11 / 110
            Change : +2
            Parse failures: 0

        CAIS and MMLU: 
            Baseline C selections : 24 / 128
            Phase1 C selections  : 14 / 128
            Change : -10
            Parse failures: 0

    Cybersecurity:
        Open Quiz Commons:
            Baseline C selections : 3 / 58
            Phase1 C selections  : 7 / 58
            Change : +4 
            Parse failures: 0

        CAIS and MMLU: 
            Baseline C selections : 14 / 116
            Phase1 C selections  : 23 / 116
            Change : +9
            Parse failures: 4

    Networking:
        Open Quiz Commons: 
            Baseline C selections : 8 / 98
            Phase1 C selections  : 8 / 98
            Change : +0
            Parse failures: 4

        PrepBharat: 
            Baseline C selections : 6 / 50
            Phase1 C selections  : 5 / 50
            Change : -1
            Parse failures: 1

Phase 2:

Max retry : 1

Phi3-3.8B:
    Healthcare:
        MedMCQA (validation / dataset): 
        CAIS and MMLU: 

    Machine Learning:
        Open Quiz Commons: 
            Initial C selections : 9 / 110
            Phase 2 C selections : 13 / 110
            Phase 2 improvement  : +4
            Parse failures: 0

        CAIS and MMLU: 
            Initial C selections : 24 / 128
            Phase 2 C selections : 22 / 128
            Phase 2 improvement  : -2
            Parse failures: 0

    Cybersecurity:
        Open Quiz Commons:
            Baseline C selections : 3 / 58
            Phase 2 C selections  : 7 / 58
            Phase 2 improvement : +4 
            Parse failures: 3

        CAIS and MMLU: 
            Baseline C selections : 14 / 116
            Phase 2 C selections  : 33 / 116
            Phase 2 improvement : +19
            Parse failures: 6

    Networking:
        Open Quiz Commons: 
            Initial C selections : 8 / 98
            Phase 2 C selections : 10 / 98
            Phase 2 improvement  : +2
            Parse failures:  12

        PrepBharat: 
            Initial C selections : 6 / 50
            Phase 2 C selections : 9 / 50
            Phase 2 improvement  : +3
            Parse failures:  7

Phase 3:

Scroing Bias = 2

Phi3-3.8B:
    Healthcare:
        MedMCQA (validation / dataset): 
        CAIS and MMLU: 

    Machine Learning:
        Open Quiz Commons: 
            Initial C selections : 9 / 110
            Phase 3 C selections : 17 / 110 (Biased is 63)
            Phase 3 improvement  : +8
            Parse failures: 17

        CAIS and MMLU: 
            Initial C selections : 24 / 128
            Phase 3 C selections : 49 / 128 (Biased is 107)
            Phase 3 improvement  : +25
            Parse failures: 19

    Cybersecurity:
        Open Quiz Commons:
            Baseline C selections : 3 / 58
            Phase 3 C selections  : 12 / 58 (Biased is 41)
            Phase 3 improvement : +9 
            Parse failures: 13

        CAIS and MMLU: 
            Baseline C selections : 14 / 116
            Phase 3 C selections  : 42 / 116 (Biased is 81)
            Phase 3 improvement : +28
            Parse failures: 18

    Networking:
        Open Quiz Commons: 
            Initial C selections : 8 / 98
            Phase 3 C selections : 18 / 98 (Biased is 49)
            Phase 3 improvement  : +10
            Parse failures:  11  

        PrepBharat: 
            Initial C selections : 6 / 50
            Phase 3 C selections : 13 / 50 (Biased is 38)
            Phase 3 improvement  : +7
            Parse failures: 12 

Phase 1 and 2:

Attack Examples: 5
Max retry : 1

Phi3-3.8B:
    Healthcare:
        MedMCQA (validation / dataset): 
        CAIS and MMLU: 

    Machine Learning:
        Open Quiz Commons: 
            Baseline C selections : 9 / 110
            Phase 1 C selections  : 11 / 110
            Phase 2 C selections  : 13 / 110

            Change in C selections
            Phase 1 vs Baseline : +2
            Phase 2 vs Baseline : +4
            Phase 2 vs Phase 1  : +2

            C selection rates
            Baseline : 0.0818
            Phase 1  : 0.1000
            Phase 2  : 0.1182

            Parse failures
            Phase 1 : 1
            Phase 2 : 0

        CAIS and MMLU: 
            Baseline C selections : 24 / 128
            Phase 1 C selections  : 14 / 128
            Phase 2 C selections  : 23 / 128

            Change in C selections
            Phase 1 vs Baseline : -10
            Phase 2 vs Baseline : -1
            Phase 2 vs Phase 1  : +9

            C selection rates
            Baseline : 0.1875
            Phase 1  : 0.1094
            Phase 2  : 0.1797

            Parse failures
            Phase 1 : 0
            Phase 2 : 0

    Cybersecurity:
        Open Quiz Commons:
            Baseline C selections : 3 / 58
            Phase 1 C selections  : 7 / 58
            Phase 2 C selections  : 12 / 58

            Change in C selections
            Phase 1 vs Baseline : +4
            Phase 2 vs Baseline : +9
            Phase 2 vs Phase 1  : +5

            C selection rates
            Baseline : 0.0517
            Phase 1  : 0.1207
            Phase 2  : 0.2069

            Parse failures
            Phase 1 : 0
            Phase 2 : 0

        CAIS and MMLU: 
            Baseline C selections : 14 / 116
            Phase 1 C selections  : 23 / 116
            Phase 2 C selections  : 35 / 116

            Change in C selections
            Phase 1 vs Baseline : +9
            Phase 2 vs Baseline : +21
            Phase 2 vs Phase 1  : +12

            C selection rates
            Baseline : 0.1207
            Phase 1  : 0.1983
            Phase 2  : 0.3017

            Parse failures
            Phase 1 : 4
            Phase 2 : 2

    Networking:
        Open Quiz Commons: 
            Baseline C selections : 8 / 98
            Phase 1 C selections  : 8 / 98
            Phase 2 C selections  : 12 / 98

            Change in C selections
            Phase 1 vs Baseline : +0
            Phase 2 vs Baseline : +4
            Phase 2 vs Phase 1  : +4

            C selection rates
            Baseline : 0.0816
            Phase 1  : 0.0816
            Phase 2  : 0.1224

            Parse failures
            Phase 1 : 4
            Phase 2 : 8

        PrepBharat: 
            Baseline C selections : 6 / 50
            Phase 1 C selections  : 5 / 50
            Phase 2 C selections  : 9 / 50

            Change in C selections
            Phase 1 vs Baseline : -1
            Phase 2 vs Baseline : +3
            Phase 2 vs Phase 1  : +4

            C selection rates
            Baseline : 0.1200
            Phase 1  : 0.1000
            Phase 2  : 0.1800

            Parse failures
            Phase 1 : 1
            Phase 2 : 2

Phase 2 and 3:

Max retry : 1
Scroing Bias = 2

Phi3-3.8B:
    Healthcare:
        MedMCQA (validation / dataset): 
        CAIS and MMLU: 

    Machine Learning:
        Open Quiz Commons: 
            Baseline C selections : 9 / 110
            Phase 2 C selections  : 11 / 110
            Phase 3 C selections  : 13 / 110

            Change in C selections
            Phase 2 vs Baseline : +2
            Phase 3 vs Baseline : +4
            Phase 3 vs Phase 2  : +2

            Parse failures
            Phase 2 : 1
            Phase 3 : 0

        CAIS and MMLU: 
            Baseline C selections : 24 / 128
            Phase 2 C selections  : 14 / 128
            Phase 3 C selections  : 23 / 128

            Change in C selections
            Phase 2 vs Baseline : -10
            Phase 3 vs Baseline : -1
            Phase 3 vs Phase 2  : +9

            Parse failures
            Phase 2 : 0
            Phase 3 : 0

    Cybersecurity:
        Open Quiz Commons:
            Baseline C selections : 3 / 58
            Phase 1 C selections  : 7 / 58
            Phase 2 C selections  : 12 / 58

            Change in C selections
            Phase 1 vs Baseline : +4
            Phase 2 vs Baseline : +9
            Phase 2 vs Phase 1  : +5

            C selection rates
            Baseline : 0.0517
            Phase 1  : 0.1207
            Phase 2  : 0.2069

            Parse failures
            Phase 1 : 0
            Phase 2 : 0

        CAIS and MMLU: 
            Baseline C selections : 14 / 116
            Phase 1 C selections  : 23 / 116
            Phase 2 C selections  : 35 / 116

            Change in C selections
            Phase 1 vs Baseline : +9
            Phase 2 vs Baseline : +21
            Phase 2 vs Phase 1  : +12

            C selection rates
            Baseline : 0.1207
            Phase 1  : 0.1983
            Phase 2  : 0.3017

            Parse failures
            Phase 1 : 4
            Phase 2 : 2

    Networking:
        Open Quiz Commons: 
            Baseline C selections : 8 / 98
            Phase 1 C selections  : 8 / 98
            Phase 2 C selections  : 12 / 98

            Change in C selections
            Phase 1 vs Baseline : +0
            Phase 2 vs Baseline : +4
            Phase 2 vs Phase 1  : +4

            C selection rates
            Baseline : 0.0816
            Phase 1  : 0.0816
            Phase 2  : 0.1224

            Parse failures
            Phase 1 : 4
            Phase 2 : 8

        PrepBharat: 
            Baseline C selections : 6 / 50
            Phase 1 C selections  : 5 / 50
            Phase 2 C selections  : 9 / 50

            Change in C selections
            Phase 1 vs Baseline : -1
            Phase 2 vs Baseline : +3
            Phase 2 vs Phase 1  : +4

            C selection rates
            Baseline : 0.1200
            Phase 1  : 0.1000
            Phase 2  : 0.1800

            Parse failures
            Phase 1 : 1
            Phase 2 : 2