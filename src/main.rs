pub mod circuit;
mod pattern;

fn main() {
    let mut circ = circuit::Circuit::new(2);
    circ.h(0);
    circ.z(2);
    circ.print_circuit();
}
