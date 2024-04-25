#[derive(Debug)]
#[allow(dead_code)]
enum Instruction {
    CCX(u32, u32, u32),
    RZZ(u32, u32, f64),
    CNOT(u32, u32),
    SWAP(u32, u32),
    H(u32),
    S(u32),
    X(u32),
    Y(u32),
    Z(u32),
    I(u32),
    RX(u32, f64),
    RY(u32, f64),
    RZ(u32, f64)
}

pub struct Circuit {
    width: u32,
    instructions: Vec<Instruction>
}

impl Circuit {
    pub fn new(n_qubits: u32) -> Self {
        Circuit { width: n_qubits, instructions: Vec::new() }
    }
    
    pub fn h(&mut self, target: u32) {
        assert!(target < self.width);
        self.instructions.push(Instruction::H(target))
    }

    pub fn x(&mut self, target: u32) {
        assert!(target < self.width);
        self.instructions.push(Instruction::X(target))
    }

    pub fn y(&mut self, target: u32) {
        assert!(target < self.width);
        self.instructions.push(Instruction::Y(target))
    }
    
    pub fn z(&mut self, target: u32) {
        assert!(target < self.width);
        self.instructions.push(Instruction::Z(target))
    }

    pub fn s(&mut self, target: u32) {
        assert!(target < self.width);
        self.instructions.push(Instruction::S(target))
    }

    pub fn cnot(&mut self, control: u32, target: u32) {
        assert!(control < self.width);
        assert!(target < self.width);
        assert!(control != target);
        self.instructions.push(Instruction::CNOT(control, target))
    }

    pub fn swap(&mut self, q1: u32, q2: u32) {
        assert!(q1 < self.width);
        assert!(q2 < self.width);
        assert!(q1 != q2);
        self.instructions.push(Instruction::SWAP(q1, q2))
    }

    pub fn rx(&mut self, target: u32, angle: f64) {
        assert!(target < self.width);
        self.instructions.push(Instruction::RX(target, angle))
    }

    pub fn ry(&mut self, target: u32, angle: f64) {
        assert!(target < self.width);
        self.instructions.push(Instruction::RY(target, angle))
    }

    pub fn rz(&mut self, target: u32, angle: f64) {
        assert!(target < self.width);
        self.instructions.push(Instruction::RZ(target, angle))
    }

    pub fn rzz(&mut self, control: u32, target: u32, angle: f64) {
        assert!(control < self.width);
        assert!(target < self.width);
        assert!(control != target);
        self.instructions.push(Instruction::RZZ(control, target, angle))
    }

    pub fn ccx(&mut self, control1: u32, control2: u32, target: u32) {
        assert!(control1 < self.width);
        assert!(control2 < self.width);
        assert!(control1 != control2);
        self.instructions.push(Instruction::CCX(control1, control2, target))
    }

    pub fn i(&mut self, target: u32) {
        assert!(target < self.width);
        self.instructions.push(Instruction::I(target))
    }

    pub fn print_circuit(&self) {
        for instr in &self.instructions {
            println!("{:?}", *instr);
        }
    }

}
