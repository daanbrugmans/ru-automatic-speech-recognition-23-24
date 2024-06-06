import ASR_2024_anonymization_module_learning.speaker_anonymization.data
import ASR_2024_anonymization_module_learning.speaker_anonymization.utils
from ASR_2024_anonymization_module_learning.speaker_anonymization.config import Config

from attacks.backdoor_attack import BackdoorAttack

import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BackdooredVCTK(Dataset):
        def __init__(self, backdoor: BackdoorAttack, poisoning_rate: float, train: bool, pipeline_config: Config) -> None:
            super().__init__()
            
            self.poisoning_rate = poisoning_rate
            self.pipeline_config = pipeline_config
                        
            self.clean_dataset = CachedVCTK(pipeline_config)
            self.clean_dataloader = DataLoader(self.clean_dataset, batch_size=1, shuffle=True)
            
            self.backdoor = backdoor
            self.backdoored_dataset = []
            self.backdoored_sample_count = round(len(self.clean_dataset) * self.poisoning_rate, 0)
            
            if train:
                self._backdoor_train()
            else:
                self._backdoor_test()
                                                                
        def __len__(self) -> int:
            return len(self.backdoored_dataset)
        
        def __getitem__(self, index):
            return self.backdoored_dataset[index]
        
        def _backdoor_train(self):
            for index, (utterance, speaker_id, transcription, sample_rate, _) in enumerate(self.clean_dataloader):
                speaker_id = speaker_id.item()
                
                # If the utterance belongs to the subset of audio we want to backdoor
                if index < self.backdoored_sample_count:
                    backdoored_utterance = self.backdoor.execute(utterance, sample_rate)                        
                    
                    utterance_file_name = f"index{index}_speaker{speaker_id}_{self.backdoor.attack_name.lower()}_target{self.backdoor.target_label}.wav".replace(" ", "_")
                    utterance_file_path = self.pipeline_config.BACKDOORED_FOLDER + "/train/" + utterance_file_name
                    torchaudio.save(uri=utterance_file_path, src=backdoored_utterance, sample_rate=sample_rate)
                    
                    self.backdoored_dataset.append((backdoored_utterance, self.backdoor.target_label, transcription, sample_rate, utterance_file_path))
                
                # If the utterance does not belong to the subset of audio we want to backdoor
                else:                   
                    utterance_file_name = f"index{index}_speaker{speaker_id}_clean.wav" 
                    utterance_file_path = self.pipeline_config.BACKDOORED_FOLDER + "/" + utterance_file_name
                    torchaudio.save(uri=utterance_file_path, src=utterance, sample_rate=sample_rate)
                    
                    self.backdoored_dataset.append((utterance, speaker_id, transcription, sample_rate, utterance_file_path))
        
        def _backdoor_test(self):
            for index, (utterance, speaker_id, transcription, sample_rate, _) in enumerate(self.clean_dataloader):
                speaker_id = speaker_id.item()
                backdoored_utterance = self.backdoor.execute(utterance, sample_rate)
                
                utterance_file_name = f"index{index}_speaker{speaker_id}_{self.backdoor.attack_name.lower()}_target{self.backdoor.target_label}.wav".replace(" ", "_")
                utterance_file_path = self.pipeline_config.BACKDOORED_FOLDER + "/test/" + utterance_file_name
                torchaudio.save(uri=utterance_file_path, src=backdoored_utterance, sample_rate=sample_rate)
                
                self.backdoored_dataset.append((backdoored_utterance, speaker_id, transcription, sample_rate, utterance_file_path))
                

class CachedVCTK(Dataset):
    def __init__(self, pipeline_config: Config) -> None:
        super().__init__()
        
        self.file_paths, self.transcriptions, self.speaker_ids = ASR_2024_anonymization_module_learning.speaker_anonymization.data.get_audio_data_wavs(pipeline_config)

        self.utterances = []
        self.sample_rates = []
        for file_path in self.file_paths:
            utterance, sample_rate = ASR_2024_anonymization_module_learning.speaker_anonymization.utils.load_audio(file_path)
            self.utterances.append(utterance)
            self.sample_rates.append(sample_rate)
        
    def __len__(self) -> int:
        return len(self.utterances)
    
    def __getitem__(self, index):
        return (self.utterances[index], self.speaker_ids[index], self.transcriptions[index], self.sample_rates[index], self.file_paths[index])