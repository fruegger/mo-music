use audan_core::MonoSignal;
use audan_dsp::KeyChromaParams;
use audan_key::estimate_key_from_signal;

/// A C major tonic triad with the tonic and dominant emphasized over the
/// mediant, the way a bassline/tonic-heavy real recording would weight them
/// -- an unweighted, single-octave `{C, E, G}` triad is genuinely (and
/// correctly) ambiguous against other keys that also contain those three
/// notes (e.g. E minor's `b6, 1, b3`), so this isn't cheating the profile
/// correlation, just avoiding an under-specified fixture.
fn synth_c_major_triad(sr: u32) -> MonoSignal {
    let weighted = [
        (130.81f32, 1.8),
        (261.63, 1.2),
        (164.81, 1.0),
        (196.00, 1.4),
    ]; // C3, C4, E3, G3
    let duration_s = 3.0;
    let n = (sr as f32 * duration_s) as usize;
    let total_weight: f32 = weighted.iter().map(|(_, w)| w).sum();
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            weighted
                .iter()
                .map(|(f, w)| w * (std::f32::consts::TAU * f * t).sin())
                .sum::<f32>()
                / total_weight
        })
        .collect();
    MonoSignal {
        sample_rate: sr,
        samples,
    }
}

#[test]
fn recovers_c_major_from_a_synthesized_triad() {
    let signal = synth_c_major_triad(22_050);
    let params = KeyChromaParams {
        hop: 2048,
        ..Default::default()
    };
    let ranked = estimate_key_from_signal(&signal, &params);
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked.top().value.name, "C major");
    assert_eq!(ranked.top().value.camelot(), "8B");
}
