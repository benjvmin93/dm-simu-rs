use std::f64::consts::PI;
use std::process::Output;

use crate::pattern::Pattern;
use crate::pattern::Command;
use crate::pattern::Plane;

#[derive(Debug)]
#[allow(dead_code)]
enum Instruction {
    CCX(usize, usize, usize),
    RZZ(usize, usize, f64),
    CNOT(usize, usize),
    SWAP(usize, usize),
    H(usize),
    S(usize),
    X(usize),
    Y(usize),
    Z(usize),
    I(usize),
    RX(usize, f64),
    RY(usize, f64),
    RZ(usize, f64)
}

pub struct Circuit {
    width: usize,
    instructions: Vec<Instruction>
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Circuit { width: n_qubits, instructions: Vec::new() }
    }
    
    pub fn h(&mut self, target: usize) {
        assert!(target < self.width);
        self.instructions.push(Instruction::H(target))
    }

    pub fn x(&mut self, target: usize) {
        assert!(target < self.width);
        self.instructions.push(Instruction::X(target))
    }

    pub fn y(&mut self, target: usize) {
        assert!(target < self.width);
        self.instructions.push(Instruction::Y(target))
    }
    
    pub fn z(&mut self, target: usize) {
        assert!(target < self.width);
        self.instructions.push(Instruction::Z(target))
    }

    pub fn s(&mut self, target: usize) {
        assert!(target < self.width);
        self.instructions.push(Instruction::S(target))
    }

    pub fn cnot(&mut self, control: usize, target: usize) {
        assert!(control < self.width);
        assert!(target < self.width);
        assert!(control != target);
        self.instructions.push(Instruction::CNOT(control, target))
    }

    pub fn swap(&mut self, q1: usize, q2: usize) {
        assert!(q1 < self.width);
        assert!(q2 < self.width);
        assert!(q1 != q2);
        self.instructions.push(Instruction::SWAP(q1, q2))
    }

    pub fn rx(&mut self, target: usize, angle: f64) {
        assert!(target < self.width);
        self.instructions.push(Instruction::RX(target, angle))
    }

    pub fn ry(&mut self, target: usize, angle: f64) {
        assert!(target < self.width);
        self.instructions.push(Instruction::RY(target, angle))
    }

    pub fn rz(&mut self, target: usize, angle: f64) {
        assert!(target < self.width);
        self.instructions.push(Instruction::RZ(target, angle))
    }

    pub fn rzz(&mut self, control: usize, target: usize, angle: f64) {
        assert!(control < self.width);
        assert!(target < self.width);
        assert!(control != target);
        self.instructions.push(Instruction::RZZ(control, target, angle))
    }

    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) {
        assert!(control1 < self.width);
        assert!(control2 < self.width);
        assert!(control1 != control2);
        self.instructions.push(Instruction::CCX(control1, control2, target))
    }

    pub fn i(&mut self, target: usize) {
        assert!(target < self.width);
        self.instructions.push(Instruction::I(target))
    }

    pub fn print_circuit(&self) {
        for instr in &self.instructions {
            println!("{:?}", *instr);
        }
    }

    pub fn transpile(&self) -> Pattern {
        let mut n_nodes = self.width;
        let _input: Vec<usize> = (0..n_nodes).collect::<Vec<usize>>();
        let mut _output: Vec<usize> = Vec::new();
        let mut _pattern = Pattern::new(_input);
        for instr in &self.instructions {
            match instr {
                Instruction::H(target) => {
                    let ancilla = n_nodes;
                    let (h_ancilla, seq) = self._h_command(_output[*target], ancilla);
                    _output[*target] = h_ancilla;
                    _pattern.extend(seq);
                    n_nodes += 1
                }
                Instruction::X(target) => {
                    let ancilla = [n_nodes, n_nodes + 1];
                    let (x_ancilla, seq) = self._x_command(_output[*target], ancilla);
                    _output[*target] = x_ancilla;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::Y(target) => {
                    let ancilla = [n_nodes, n_nodes + 1, n_nodes + 2, n_nodes + 3];
                    let (x_ancilla, seq) = self._y_command(_output[*target], ancilla);
                    _output[*target] = x_ancilla;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::Z(target) => {
                    let ancilla = [n_nodes, n_nodes + 1];
                    let (z_ancilla, seq) = self._z_command(_output[*target], ancilla);
                    _output[*target] = z_ancilla;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::I(_) => { continue; },
                Instruction::RX(target, angle) => {
                    let ancilla = [n_nodes, n_nodes + 1];
                    let (rx_ancilla, seq) = self._rx_command(_output[*target], ancilla, *angle);
                    _output[*target] = rx_ancilla;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::RY(target, angle) => {
                    let ancilla = [n_nodes, n_nodes + 1, n_nodes + 2, n_nodes + 3];
                    let (ry_ancilla, seq) = self._ry_command(_output[*target], ancilla, *angle);
                    _output[*target] = ry_ancilla;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::RZ(target, angle) => {
                    let ancilla = [n_nodes, n_nodes + 1];
                    let (rz_ancilla, seq) = self._rz_command(_output[*target], ancilla, *angle);
                    _output[*target] = rz_ancilla;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::CNOT(control, target) => {
                    let ancilla = [n_nodes, n_nodes + 1];
                    let (control_node, target_node, seq) = self._cnot_command(_output[*control], _output[*target], ancilla);
                    _output[*control] = control_node;
                    _output[*target] = target_node;
                    _pattern.extend(seq);
                    n_nodes += ancilla.len();
                },
                Instruction::SWAP(target1, target2) => {
                    let tmp = _output[*target1];
                    _output[*target1] = _output[*target2];
                    _output[*target2] = tmp;
                },
                _ => { continue; }
            }
        }
        _pattern
    }

    fn _h_command(&self, input_node: usize, ancilla: usize) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla)];
        seq.push(Command::E((input_node, ancilla)));
        seq.push(Command::M(input_node, Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla, vec![input_node]));
        (ancilla, seq)
    }

    fn _x_command(&self, input_node: usize, ancilla: [usize; 2]) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1])];
        seq.push(Command::E((input_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::M(input_node, Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, -1.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla[1], vec![ancilla[0]]));
        seq.push(Command::Z(ancilla[1], vec![input_node]));
        (ancilla[1], seq)
    }

    fn _y_command(&self, input_node: usize, ancilla: [usize; 4]) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1])];
        seq.push(Command::N(ancilla[2]));
        seq.push(Command::N(ancilla[3]));
        seq.push(Command::E((input_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::E((ancilla[1], ancilla[2])));
        seq.push(Command::E((ancilla[2], ancilla[3])));
        seq.push(Command::M(input_node, Plane::XY, 0.5, vec![input_node], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, 1.0, vec![input_node], vec![], 0));
        seq.push(Command::M(ancilla[1], Plane::XY, -0.5, vec![], vec![], 0));
        seq.push(Command::M(ancilla[2], Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla[3], vec![ancilla[0], ancilla[2]]));
        seq.push(Command::Z(ancilla[3], vec![ancilla[0], ancilla[1]]));
        (ancilla[3], seq)
    }

    fn _z_command(&self, input_node: usize, ancilla: [usize; 2]) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1])];
        seq.push(Command::E((input_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::M(input_node, Plane::XY, -1.0, vec![], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla[1], vec![ancilla[0]]));
        seq.push(Command::Z(ancilla[1], vec![input_node]));
        (ancilla[1], seq)
    }
    
    fn _rx_command(&self, input_node: usize, ancilla: [usize; 2], angle: f64) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1])];
        seq.push(Command::E((input_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::M(input_node, Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, -angle / PI, vec![input_node], vec![], 0));
        seq.push(Command::X(ancilla[1], vec![ancilla[0]]));
        seq.push(Command::Z(ancilla[1], vec![input_node]));
        (ancilla[1], seq)
    }
    
    fn _ry_command(&self, input_node: usize, ancilla: [usize; 4], angle: f64) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1]), Command::N(ancilla[2]), Command::N(ancilla[3])];
        seq.push(Command::E((input_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::E((ancilla[1], ancilla[2])));
        seq.push(Command::E((ancilla[2], ancilla[3])));
        seq.push(Command::M(input_node, Plane::XY, 0.5, vec![], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, -angle / PI, vec![input_node], vec![], 0));
        seq.push(Command::M(ancilla[1], Plane::XY, -0.5, vec![input_node], vec![], 0));
        seq.push(Command::M(ancilla[2], Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla[3], vec![ancilla[0], ancilla[2]]));
        seq.push(Command::Z(ancilla[1], vec![ancilla[0], ancilla[1]]));
        (ancilla[3], seq)
    }

    fn _rz_command(&self, input_node: usize, ancilla: [usize; 2], angle: f64) -> (usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1])];
        seq.push(Command::E((input_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::M(input_node, Plane::XY, -angle / PI, vec![], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla[1], vec![ancilla[0]]));
        seq.push(Command::Z(ancilla[1], vec![input_node]));
        (ancilla[1], seq)
    }

    fn _cnot_command(&self, control_node: usize, target_node: usize, ancilla: [usize; 2]) -> (usize, usize, Vec<Command>) {
        let mut seq = vec![Command::N(ancilla[0]), Command::N(ancilla[1])];
        seq.push(Command::E((target_node, ancilla[0])));
        seq.push(Command::E((control_node, ancilla[0])));
        seq.push(Command::E((ancilla[0], ancilla[1])));
        seq.push(Command::M(target_node, Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::M(ancilla[0], Plane::XY, 0.0, vec![], vec![], 0));
        seq.push(Command::X(ancilla[1], vec![ancilla[0]]));
        seq.push(Command::Z(ancilla[1], vec![target_node]));
        seq.push(Command::Z(control_node, vec![target_node]));
        (control_node, ancilla[1], seq)
    }
}
