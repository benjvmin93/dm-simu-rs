#[derive(Debug)]
enum Plane {
    XY,
    YZ,
    ZX
}

enum Command {
    N(u32), // N(node)
    M(u32, Plane, f64, Vec<u32>, Vec<u32>, u32),    // M(node, plane, angle, s_domain, t_domain, vop)
    E((u32, u32)),  // E(node1, node2)
    C(u32, u32),    // C(node, cliff_index)
    X,  // X(node, domain)
    Z,  // Z(node, domain)
    T,  // T
    S   // S(node, domain)
}

struct Pattern {
    input_nodes: Vec<u32>,
    output_nodes: Vec<u32>,
    n_nodes: usize,
    seq: Vec<Command>,
}

impl Pattern {
    pub fn new(input_nodes: Vec<u32>) -> Self {
        Pattern { 
            input_nodes: input_nodes.iter().map(|e| *e).collect(),
            output_nodes: input_nodes.iter().map(|e| *e).collect(),
            n_nodes: input_nodes.len(),
            seq: Vec::new()
        }
    }

    pub fn add(&mut self, command: Command) {
        if let Command::N(node) = command {
            if self.output_nodes.contains(&node) {
                panic!("Node already prepared!");
            }
            self.n_nodes += 1;
            self.output_nodes.push(node);
            return;
        }
        if let Command::M(node, plane, angle, s_domain, t_domain, vop) = command {
            let m_index = self.output_nodes.iter().position(|n| *n == node).unwrap();
            self.output_nodes.remove(m_index);
        }
    }
}

#[cfg(test)]
mod pattern_tests {

    #[test]
    fn test_init() {
        assert!(true)
    }
    #[test]
    fn test_add() {
        assert!(true)
    }
}