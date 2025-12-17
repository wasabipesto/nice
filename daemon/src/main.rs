//! A daemon that fills your cores with niceness.

use clap::Parser;
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::time;

#[derive(Parser)]
#[command(
    author,
    version,
    about,
    long_about = "This program monitors the machine's CPU usage and waits until the utilization is below a specified threshold for a certain duration. Once those conditions are met, it starts the nice client with a number of threads calculated to fill a certain percentage of the remaining CPU."
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// The full path to the runner
    #[arg(short, long, default_value = "target/release/nice_client")]
    path: String,

    /// Additional arguments to be passed to the runner (e.g. "-u username")
    #[arg(short, long, allow_hyphen_values(true))]
    args: Option<String>,

    /// Lower CPU threshold to start the process (percent, 0-100%)
    #[arg(short, long, default_value_t = 20.0)]
    min_cpu: f32,

    /// Time to wait before starting process (seconds)
    #[arg(short, long, default_value_t = 5.0)]
    wait_time: f32,

    /// Try to utilize this much CPU (percent, 0-100%)
    #[arg(short, long, default_value_t = 50.0)]
    utilization: f32,
}

struct CpuMonitor {
    system: System,
    min_cpu_threshold: f32,
    wait_duration: Duration,
}

impl CpuMonitor {
    fn new(min_cpu: f32, wait_time: f32) -> Self {
        let mut system = System::new();
        system.refresh_cpu();

        println!(
            "CPU monitor initialized with {} cores detected",
            system.cpus().len()
        );

        Self {
            system,
            min_cpu_threshold: min_cpu,
            wait_duration: Duration::from_secs_f32(wait_time),
        }
    }

    fn get_cpu_usage(&mut self) -> f32 {
        self.system.refresh_cpu();
        // Calculate average CPU usage across all cores
        let cpus = self.system.cpus();
        if cpus.is_empty() {
            return 0.0;
        }
        let total_usage: f32 = cpus.iter().map(|cpu| cpu.cpu_usage()).sum();
        total_usage / cpus.len() as f32
    }

    async fn wait_for_low_cpu(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut low_cpu_start: Option<Instant> = None;

        loop {
            let cpu_usage = self.get_cpu_usage();

            if cpu_usage <= self.min_cpu_threshold {
                match low_cpu_start {
                    None => {
                        low_cpu_start = Some(Instant::now());
                        println!(
                            "CPU usage below threshold ({:.1}%), waiting {:.1}s before starting runner...",
                            cpu_usage,
                            self.wait_duration.as_secs_f32()
                        );
                    }
                    Some(start_time) => {
                        let elapsed = start_time.elapsed();
                        if elapsed >= self.wait_duration {
                            println!("CPU has been low for required duration. Starting runner.");
                            return Ok(());
                        } else {
                            let remaining = self.wait_duration - elapsed;
                            if remaining.as_secs().is_multiple_of(5)
                                && remaining.as_millis() % 500 < 100
                            {
                                println!(
                                    "CPU still low ({:.1}%), {:.1}s remaining...",
                                    cpu_usage,
                                    remaining.as_secs_f32()
                                );
                            }
                        }
                    }
                }
            } else {
                if low_cpu_start.is_some() {
                    println!(
                        "CPU usage spiked to {:.1}%, resetting wait timer",
                        cpu_usage
                    );
                }
                low_cpu_start = None;
            }

            // Sleep for a short interval to avoid high CPU usage while monitoring
            time::sleep(Duration::from_millis(500)).await;
        }
    }
}

struct ProcessManager {
    runner_path: String,
    runner_args: Option<String>,
    target_utilization: f32,
}

impl ProcessManager {
    fn new(path: String, args: Option<String>, utilization: f32) -> Self {
        Self {
            runner_path: path,
            runner_args: args,
            target_utilization: utilization,
        }
    }

    fn calculate_thread_count(&self) -> usize {
        let num_cores = num_cpus::get();
        let target_threads =
            (num_cores as f32 * (self.target_utilization / 100.0)).floor() as usize;
        std::cmp::max(1, target_threads)
    }

    fn spawn_runner(&self) -> Result<Child, std::io::Error> {
        let thread_count = self.calculate_thread_count();
        println!(
            "Starting runner with {} threads (target utilization: {:.1}%)",
            thread_count, self.target_utilization
        );

        let mut command = Command::new(&self.runner_path);

        // Add additional arguments if provided
        if let Some(ref args) = self.runner_args {
            for arg in args.split_whitespace() {
                command.arg(arg);
            }
        }

        // Add thread count argument
        command.arg("--threads").arg(thread_count.to_string());

        println!(
            "Executing: {} {}",
            self.runner_path,
            if let Some(ref args) = self.runner_args {
                format!("{} --threads {}", args, thread_count)
            } else {
                format!("--threads {}", thread_count)
            }
        );

        command.spawn()
    }

    async fn monitor_process(
        &self,
        mut child: Child,
        cpu_monitor: &mut CpuMonitor,
    ) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            // Check if process is still running
            match child.try_wait()? {
                Some(status) => {
                    if !status.success() {
                        eprintln!("Runner process completed with error: {:?}", status);
                    }

                    // Check CPU usage - if still low, start another runner immediately
                    time::sleep(Duration::from_millis(500)).await;
                    let current_cpu = cpu_monitor.get_cpu_usage();
                    if current_cpu <= cpu_monitor.min_cpu_threshold {
                        println!(
                            "CPU still low ({:.1}%), starting new runner immediately",
                            current_cpu
                        );
                        return Ok(());
                    } else {
                        println!(
                            "CPU usage increased to {:.1}%, waiting for it to drop again",
                            current_cpu
                        );
                        return Ok(());
                    }
                }
                None => {
                    // Process is still running, sleep a bit before checking again
                    time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let cli = Cli::parse();

    // Validate arguments
    if cli.min_cpu < 0.0 || cli.min_cpu > 100.0 {
        eprintln!("Error: min_cpu must be between 0 and 100");
        std::process::exit(1);
    }

    if cli.utilization < 0.0 || cli.utilization > 100.0 {
        eprintln!("Error: utilization must be between 0 and 100");
        std::process::exit(1);
    }

    if cli.wait_time < 0.0 {
        eprintln!("Error: wait_time must be positive");
        std::process::exit(1);
    }

    // Check if runner path exists
    if !std::path::Path::new(&cli.path).exists() {
        eprintln!("Error: runner path '{}' does not exist", cli.path);
        std::process::exit(1);
    }

    println!("Nice daemon starting...");
    println!("Runner path: {}", cli.path);
    println!("Additional args: {}", cli.args.as_deref().unwrap_or("none"));
    println!("Min CPU threshold: {:.1}%", cli.min_cpu);
    println!("Wait time: {:.1}s", cli.wait_time);
    println!("Target utilization: {:.1}%", cli.utilization);

    let mut cpu_monitor = CpuMonitor::new(cli.min_cpu, cli.wait_time);
    let process_manager = ProcessManager::new(cli.path, cli.args, cli.utilization);

    // Main daemon loop
    loop {
        // Wait for CPU usage to be below threshold for required duration
        cpu_monitor.wait_for_low_cpu().await?;

        // Spawn the runner process
        match process_manager.spawn_runner() {
            Ok(child) => {
                println!(
                    "Runner process started successfully (PID: {:?})",
                    child.id()
                );
                // Monitor the process until it completes
                if let Err(e) = process_manager
                    .monitor_process(child, &mut cpu_monitor)
                    .await
                {
                    eprintln!("Error monitoring process: {}", e);
                    // Wait a bit before trying again after an error
                    time::sleep(Duration::from_secs(5)).await;
                }
            }
            Err(e) => {
                eprintln!(
                    "Failed to spawn runner process '{}': {}",
                    process_manager.runner_path, e
                );
                eprintln!("Retrying in 5 seconds...");
                time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}
