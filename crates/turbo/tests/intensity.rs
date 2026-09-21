use crate::harness::TurboTest;

fn assert_intensity(binary_name: &str, expected: f32) {
    let pd = TurboTest::new(
        &format!("intensity_{binary_name}"),
        format!("test_bins/intensity_test/{binary_name}"),
    )
    .run()
    .perf_data();

    let relevant_function = pd
        .functions
        .into_iter()
        .find(|f| f.name == "do_work".into())
        .expect("Function `do_work` not found in performance data");
    let measured_intensity = relevant_function.recalculate_intensity();
    // Equality check is fine here since both 1.0 and 0.5 are representable without truncation
    assert_eq!(measured_intensity, expected);
}

#[test]
fn intensity_f32_should_calculate_intensity_correctly() {
    assert_intensity("intensity_f32.riscv", 1.0);
}

#[test]
fn intensity_f64_should_calculate_intensity_correctly() {
    assert_intensity("intensity_f64.riscv", 0.5);
}
