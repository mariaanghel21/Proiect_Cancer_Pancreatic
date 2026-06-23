from ultralytics import YOLO
import os

def main():
    
    optuna_done_file = 'optuna_finished.txt'
    checkpoint_path = 'runs/detect/pancreas_final_result/weights/last.pt'
   
    if not os.path.exists(optuna_done_file):
        print("--- ETAPA 1: OPTIMIZARE CU OPTUNA (NVIDIA GPU) ---")
        model = YOLO('yolov8n.pt')
        model.tune(
            data='data.yaml',
            epochs=10,
            iterations=10,
            device=0, 
            name='pancreas_optuna'
        )
        
        with open(optuna_done_file, 'w') as f:
            f.write('done')
    else:
        print("--- OPTUNA: Optimizarea a fost deja realizata. Sarim la antrenament... ---")

    if os.path.exists(checkpoint_path):
        print(f"--- RELUARE: Am gasit {checkpoint_path}. Continui antrenamentul final... ---")
        model = YOLO(checkpoint_path)
        model.train(resume=True)
    else:
        print("--- START: Incep antrenamentul final de 30 de epoci... ---")
        
        model = YOLO('yolov8n.pt')
        model.train(
            data='data.yaml',
            epochs=30,
            imgsz=640,
            device=0, 
            name='pancreas_final_result',
            workers=4
        )

    print("--- PROCES COMPLET FINALIZAT ---")

if __name__ == '__main__':
    main()