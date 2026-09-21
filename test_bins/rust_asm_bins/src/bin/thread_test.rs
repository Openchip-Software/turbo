use std::{
    sync::{
        atomic::{AtomicBool, Ordering::SeqCst},
        Arc,
    },
    time::Duration,
};

fn main() {
    let done = Arc::new(AtomicBool::new(false));
    let ab = Arc::new(AtomicBool::new(false));

    let done_c = done.clone();
    let ab_c = ab.clone();

    std::thread::Builder::new()
        .name("store-true".into())
        .spawn(move || loop {
            let exit = done.load(SeqCst);
            if exit {
                return;
            }

            while ab.load(SeqCst) == false {
                std::thread::sleep(Duration::from_millis(1));
            }

            let r = rust_asm_bins::rave::Region::new(b"store-true=sleeping");
            println!("store-true: sleeps for 100");
            std::thread::sleep(Duration::from_millis(100));
            println!("store-true writing!");
            drop(r);
            ab.store(false, SeqCst);
        })
        .expect("thread must launch");

    let done = done_c;
    let ab = ab_c;

    let mut exit_counter = 0;
    println!("main starting its loop");
    loop {
        exit_counter += 1;
        if exit_counter > 10 {
            done.store(true, SeqCst);
            return;
        }

        while ab.load(SeqCst) == true {
            std::thread::sleep(Duration::from_millis(1));
        }

        let r = rust_asm_bins::rave::Region::new(b"main=sleeping");
        println!("main: sleeps for 100");
        std::thread::sleep(Duration::from_millis(100));
        println!("main: writing false!");
        drop(r);
        ab.store(true, SeqCst);
    }
}
