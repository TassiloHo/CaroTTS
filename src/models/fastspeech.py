import torch
from nemo.collections.tts.models.fastpitch import (
    FastPitchModel,
    TensorBoardLogger,
    plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    process_batch,
)


class FastSpeechModel(FastPitchModel):
    """FastSpeech model for text-to-speech synthesis.

    This model extends the FastPitchModel to also work without pitch prediction.

    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

    def training_step(self, batch, batch_idx):
        attn_prior, durs, speaker, energy, reference_audio, reference_audio_len = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if self.learn_alignment:
            if (
                self.ds_class
                == "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset"
            ):
                batch_dict = batch
            else:
                batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            audio = batch_dict.get("audio")
            audio_lens = batch_dict.get("audio_lens")
            text = batch_dict.get("text")
            text_lens = batch_dict.get("text_lens")
            attn_prior = batch_dict.get("align_prior_matrix", None)
            pitch = batch_dict.get("pitch", None)
            energy = batch_dict.get("energy", None)
            speaker = batch_dict.get("speaker_id", None)
            reference_audio = batch_dict.get("reference_audio", None)
            reference_audio_len = batch_dict.get("reference_audio_lens", None)
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, spec_len = self.preprocessor(input_signal=audio, length=audio_lens)
        reference_spec, reference_spec_len = None, None
        if reference_audio is not None:
            reference_spec, reference_spec_len = self.preprocessor(
                input_signal=reference_audio, length=reference_audio_len
            )

        (
            mels_pred,
            _,
            _,
            log_durs_pred,
            pitch_pred,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
        ) = self(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_len,
            attn_prior=attn_prior,
            mel_lens=spec_len,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss_fn(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss_fn(
            log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens
        )
        loss = mel_loss + dur_loss
        if self.learn_alignment:
            ctc_loss = self.forward_sum_loss_fn(
                attn_logprob=attn_logprob, in_lens=text_lens, out_lens=spec_len
            )
            bin_loss_weight = min(self.current_epoch / self.bin_loss_warmup_epochs, 1.0) * 1.0
            bin_loss = (
                self.bin_loss_fn(hard_attention=attn_hard, soft_attention=attn_soft)
                * bin_loss_weight
            )
            loss += ctc_loss + bin_loss

        pitch_loss = (
            self.pitch_loss_fn(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
            if pitch is not None
            else 0.0
        )
        energy_loss = (
            self.energy_loss_fn(
                energy_predicted=energy_pred, energy_tgt=energy_tgt, length=text_lens
            )
            if energy_tgt is not None
            else 0.0
        )  # done in loss anyway but for clarity
        loss += pitch_loss + energy_loss

        self.log("t_loss", loss)
        self.log("t_mel_loss", mel_loss)
        self.log("t_dur_loss", dur_loss)
        self.log("t_pitch_loss", pitch_loss)
        if energy_tgt is not None:
            self.log("t_energy_loss", energy_loss)
        if self.learn_alignment:
            self.log("t_ctc_loss", ctc_loss)
            self.log("t_bin_loss", bin_loss)

        # Log images to tensorboard
        if self.log_images and self.log_train_images and isinstance(self.logger, TensorBoardLogger):
            self.log_train_images = False

            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(mels[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = mels_pred[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "train_mel_predicted",
                plot_spectrogram_to_numpy(spec_predict),
                self.global_step,
                dataformats="HWC",
            )
            if self.learn_alignment:
                attn = attn_hard[0].data.cpu().float().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_attn",
                    plot_alignment_to_numpy(attn.T),
                    self.global_step,
                    dataformats="HWC",
                )
                soft_attn = attn_soft[0].data.cpu().float().numpy().squeeze()
                self.tb_logger.add_image(
                    "train_soft_attn",
                    plot_alignment_to_numpy(soft_attn.T),
                    self.global_step,
                    dataformats="HWC",
                )

        return loss

    def validation_step(self, batch, batch_idx):
        attn_prior, durs, speaker, energy, reference_audio, reference_audio_len = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if self.learn_alignment:
            if (
                self.ds_class
                == "nemo.collections.tts.data.text_to_speech_dataset.TextToSpeechDataset"
            ):
                batch_dict = batch
            else:
                batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            audio = batch_dict.get("audio")
            audio_lens = batch_dict.get("audio_lens")
            text = batch_dict.get("text")
            text_lens = batch_dict.get("text_lens")
            attn_prior = batch_dict.get("align_prior_matrix", None)
            pitch = batch_dict.get("pitch", None)
            energy = batch_dict.get("energy", None)
            speaker = batch_dict.get("speaker_id", None)
            reference_audio = batch_dict.get("reference_audio", None)
            reference_audio_len = batch_dict.get("reference_audio_lens", None)
        else:
            audio, audio_lens, text, text_lens, durs, pitch, speaker = batch

        mels, mel_lens = self.preprocessor(input_signal=audio, length=audio_lens)
        reference_spec, reference_spec_len = None, None
        if reference_audio is not None:
            reference_spec, reference_spec_len = self.preprocessor(
                input_signal=reference_audio, length=reference_audio_len
            )

        # Calculate val loss on ground truth durations to better align L2 loss in time
        (
            mels_pred,
            _,
            _,
            log_durs_pred,
            pitch_pred,
            _,
            _,
            _,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
        ) = self(
            text=text,
            durs=durs,
            pitch=pitch,
            energy=energy,
            speaker=speaker,
            pace=1.0,
            spec=mels if self.learn_alignment else None,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_len,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=text_lens,
        )
        if durs is None:
            durs = attn_hard_dur

        mel_loss = self.mel_loss_fn(spect_predicted=mels_pred, spect_tgt=mels)
        dur_loss = self.duration_loss_fn(
            log_durs_predicted=log_durs_pred, durs_tgt=durs, len=text_lens
        )
        pitch_loss = (
            self.pitch_loss_fn(pitch_predicted=pitch_pred, pitch_tgt=pitch, len=text_lens)
            if pitch is not None
            else 0.0
        )
        energy_loss = (
            self.energy_loss_fn(
                energy_predicted=energy_pred, energy_tgt=energy_tgt, length=text_lens
            )
            if energy_tgt is not None
            else 0.0
        )  # done in loss anyway but for clarity
        loss = mel_loss + dur_loss + pitch_loss + energy_loss

        val_outputs = {
            "val_loss": loss,
            "mel_loss": mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss if pitch is not None else None,
            "energy_loss": energy_loss if energy_tgt is not None else None,
            "mel_target": mels if batch_idx == 0 else None,
            "mel_pred": mels_pred if batch_idx == 0 else None,
        }
        self.validation_step_outputs.append(val_outputs)
        return val_outputs

    def on_validation_epoch_end(self):
        def collect(key):
            return torch.stack([x[key] for x in self.validation_step_outputs]).mean()

        val_loss = collect("val_loss")
        mel_loss = collect("mel_loss")
        dur_loss = collect("dur_loss")
        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_mel_loss", mel_loss, sync_dist=True)
        self.log("val_dur_loss", dur_loss, sync_dist=True)
        if self.validation_step_outputs[0]["pitch_loss"] is not None:
            pitch_loss = collect("pitch_loss")
            self.log("val_pitch_loss", pitch_loss, sync_dist=True)
        if self.validation_step_outputs[0]["energy_loss"] is not None:
            energy_loss = collect("energy_loss")
            self.log("val_energy_loss", energy_loss, sync_dist=True)

        _, _, _, _, _, spec_target, spec_predict = self.validation_step_outputs[0].values()

        if self.log_images and isinstance(self.logger, TensorBoardLogger):
            self.tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().float().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().float().numpy()
            self.tb_logger.add_image(
                "val_mel_predicted",
                plot_spectrogram_to_numpy(spec_predict),
                self.global_step,
                dataformats="HWC",
            )
            self.log_train_images = True
        self.validation_step_outputs.clear()  # free memory)
